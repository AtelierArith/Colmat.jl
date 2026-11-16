"""Regression checks for scope and stale/partial execution evidence."""
from pathlib import Path
import copy
import hashlib
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbt_execution import SOURCE_FILES, execution_fields, registration_digest, validate_execution


class ExecutionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = [dict(symbol=symbol, interface='Fortran', family=family, scalar='d',
                          adapter_lp64='true', adapter_ilp64='true')
                     for symbol, family in (('dgetrf_', 'getrf'), ('dlartg_', 'lartg'), ('dgemm_', 'gemm'))]
        self.data = dict(schema_version=1, full_suite_passed=True, errors=[], revision='a' * 40,
                         native_revision='b' * 40, julia_version='1.13.3', platform='fixture-platform', library_sha256='c' * 64, inventory_registration_sha256=registration_digest(self.rows),
                         source_sha256={}, witnesses=[])
        (self.root / 'Manifest.toml').write_text('manifest = "fixture"\n')
        self.data['manifest_sha256'] = hashlib.sha256((self.root / 'Manifest.toml').read_bytes()).hexdigest()
        for file in SOURCE_FILES:
            path = self.root / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture\n')
            self.data['source_sha256'][file] = hashlib.sha256(path.read_bytes()).hexdigest()
        for symbol in ('dgetrf_', 'dlartg_'):
            for bits in (32, 64):
                self.data['witnesses'].append(dict(symbol=symbol, bits=bits, route='lbt_trampoline',
                    file='test/suite.jl', line=1, signature=f'(Ref{{Int{bits}}},)' if symbol=='dgetrf_' else '(Ref{Float64},)', integer_arguments=symbol=='dgetrf_'))

    def validate(self, data=None):
        return validate_execution(self.root, self.rows, 'b' * 40, self.data if data is None else data)

    def test_scope_and_integer_free_calls(self):
        executed = self.validate()
        self.assertEqual(set(executed), {'dgetrf_', 'dlartg_'})
        self.assertEqual(execution_fields('dgemm_', executed, self.data)['execution_evidence_status'], 'not_catalogued')
        self.assertEqual(execution_fields('dlartg_', executed, self.data)['execution_evidence_status'], 'direct_call_witness')

    def test_registration_is_not_execution(self):
        self.data['witnesses'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing executed integer ABI'):
            self.validate()

    def test_failed_or_instrumentation_error_is_rejected(self):
        for field, value in (('full_suite_passed', False), ('errors', ['instrumentation error'])):
            data = copy.deepcopy(self.data)
            data[field] = value
            with self.assertRaises(ValueError):
                self.validate(data)

    def test_stale_native_inventory_manifest_and_source(self):
        for field in ('native_revision', 'inventory_registration_sha256', 'manifest_sha256'):
            data = copy.deepcopy(self.data)
            data[field] = 'stale'
            with self.assertRaises(ValueError):
                self.validate(data)
        (self.root / 'test/suite.jl').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Stale execution evidence source'):
            self.validate()

    def test_production_and_worker_changes_invalidate_evidence(self):
        native = self.root / 'deps/rust/src/lib.rs'
        original = native.read_text()
        native.write_text('// changed implementation\n')
        with self.assertRaisesRegex(ValueError, 'Stale execution evidence source'):
            self.validate()
        native.write_text(original)
        (self.root / 'test/new_worker.jl').write_text('# newly added worker\n')
        with self.assertRaisesRegex(ValueError, 'omits current production'):
            self.validate()

    def test_incomplete_integer_signature_and_unknown_route(self):
        for field, value in (('integer_arguments', False), ('route', 'registration_only'), ('signature', ''), ('signature', '(Ref{Int64},)')):
            data = copy.deepcopy(self.data)
            data['witnesses'][0][field] = value
            with self.assertRaises(ValueError):
                self.validate(data)

    def test_execution_columns_do_not_invalidate_registration(self):
        digest = registration_digest(self.rows)
        self.rows[0]['execution_evidence_status'] = 'direct_call_witness'
        self.assertEqual(registration_digest(self.rows), digest)

    def add_cblas(self):
        for symbol, family in (('cblas_dgemm', 'gemm'), ('cblas_drotg', 'rotg')):
            self.rows.append(dict(symbol=symbol, interface='CBLAS', family=family, scalar='d',
                                  adapter_lp64='true', adapter_ilp64='true'))
            for bits in (32, 64):
                signature = (f'(Int32, Int32, Int32, Int{bits}, Int{bits}, Int{bits}, Float64, '
                             f'Ptr{{Float64}}, Int{bits}, Ptr{{Float64}}, Int{bits}, Float64, Ptr{{Float64}}, Int{bits})'
                             if family == 'gemm' else '(Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64})')
                for route in ('dll', 'lbt_trampoline'):
                    self.data['witnesses'].append(dict(symbol=symbol, bits=bits, route=route,
                        file='test/suite.jl', line=1, signature=signature, integer_arguments=family!='rotg'))
        self.data['cblas_target_symbols'] = ['cblas_dgemm', 'cblas_drotg']
        self.data['inventory_registration_sha256'] = registration_digest(self.rows)

    def test_cblas_enums_and_dimension_free_calls(self):
        self.add_cblas()
        executed = self.validate()
        self.assertEqual(len(executed['cblas_dgemm']), 4)
        self.assertEqual(len(executed['cblas_drotg']), 4)

    def test_cblas_requires_both_actual_routes(self):
        self.add_cblas()
        self.data['witnesses'] = [w for w in self.data['witnesses'] if not (
            w['symbol']=='cblas_dgemm' and w['bits']==64 and w['route']=='lbt_trampoline')]
        with self.assertRaisesRegex(ValueError, 'Missing completed CBLAS ABI/route'):
            self.validate()

    def test_cblas_rejects_wrong_enum_or_dimension_abi(self):
        self.add_cblas()
        index = next(i for i,w in enumerate(self.data['witnesses']) if w['symbol']=='cblas_dgemm' and w['bits']==64)
        original = self.data['witnesses'][index]['signature']
        for signature in (original.replace('Int32', 'Int64', 1),
                          original.replace('Int64', 'Int32'),
                          original.replace('Int64', 'Ref{Int64}', 1)):
            data = copy.deepcopy(self.data)
            data['witnesses'][index]['signature'] = signature
            with self.assertRaisesRegex(ValueError, 'CBLAS enum/dimension'):
                self.validate(data)

    def test_cblas_rejects_registration_only_witness_and_changed_scope(self):
        self.add_cblas()
        source = self.root / 'test/cblas.jl'
        source.write_text('# registration-only fixture\n')
        self.data['source_sha256']['test/cblas.jl'] = hashlib.sha256(source.read_bytes()).hexdigest()
        index = next(i for i,w in enumerate(self.data['witnesses']) if w['symbol']=='cblas_dgemm')
        data = copy.deepcopy(self.data)
        data['witnesses'][index]['file'] = 'test/cblas.jl'
        with self.assertRaisesRegex(ValueError, 'registration/recorder probe'):
            self.validate(data)
        self.data['cblas_target_symbols'].pop()
        with self.assertRaisesRegex(ValueError, 'scope differs'):
            self.validate()

    def add_lapacke(self):
        declarations = {
            'LAPACKE_dgetri': ('getri', 'd', lambda j: f'(Int32, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_dgetri_work': ('getri', 'd', lambda j: f'(Int32, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zpotrs': ('potrs', 'z', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zsyr': ('syr', 'z', lambda j: f'(Int32, Int8, {j}, ComplexF64, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgesvd': ('gesvd', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgesvd_work': ('gesvd', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zgesvd': ('gesvd', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zgesvd_work': ('gesvd', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgesdd': ('gesdd', 'd', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgesdd_work': ('gesdd', 'd', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_zgesdd': ('gesdd', 'z', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgesdd_work': ('gesdd', 'z', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{{j}}})'),
            'LAPACKE_dsyev': ('syev', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dsyev_work': ('syev', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zheev': ('heev', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zheev_work': ('heev', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
        }
        for symbol, (family, scalar, signature) in declarations.items():
            self.rows.append(dict(symbol=symbol, interface='LAPACKE', family=family, scalar=scalar,
                                  adapter_lp64='true', adapter_ilp64='true'))
            for bits in (32, 64):
                for route in ('dll', 'lbt_trampoline'):
                    self.data['witnesses'].append(dict(symbol=symbol, bits=bits, route=route,
                        file='test/suite.jl', line=1, signature=signature(f'Int{bits}'),
                        return_type=f'Int{bits}', integer_arguments=True))
        self.data['lapacke_target_symbols'] = sorted(declarations)
        self.data['inventory_registration_sha256'] = registration_digest(self.rows)

    def test_lapacke_exact_signatures_and_returns(self):
        self.add_lapacke()
        result = self.validate()
        self.assertEqual(len(result['LAPACKE_zpotrs']), 4)
        self.assertEqual(len(result['LAPACKE_dgetri_work']), 4)

    def test_lapacke_rejects_layout_flag_dimension_pointer_and_return_mismatches(self):
        self.add_lapacke()
        index = next(i for i, w in enumerate(self.data['witnesses'])
                     if w['symbol']=='LAPACKE_zpotrs' and w['bits']==64)
        original = self.data['witnesses'][index]['signature']
        for signature in (original.replace('Int32', 'Int64', 1),
                          original.replace('Int8', 'Int32', 1),
                          original.replace('Int64', 'Ref{Int64}', 1),
                          original.replace('ComplexF64', 'Float64')):
            data = copy.deepcopy(self.data)
            data['witnesses'][index]['signature'] = signature
            with self.assertRaisesRegex(ValueError, 'LAPACKE layout/flag/integer/return'):
                self.validate(data)
        data = copy.deepcopy(self.data)
        data['witnesses'][index]['return_type'] = 'Int32'
        with self.assertRaisesRegex(ValueError, 'LAPACKE layout/flag/integer/return'):
            self.validate(data)

    def test_lapacke_complex_syr_coefficient_is_by_value(self):
        self.add_lapacke()
        index=next(i for i,w in enumerate(self.data['witnesses']) if w['symbol']=='LAPACKE_zsyr')
        self.data['witnesses'][index]['signature']=self.data['witnesses'][index]['signature'].replace('ComplexF64','Ptr{ComplexF64}',1)
        with self.assertRaisesRegex(ValueError,'LAPACKE layout/flag/integer/return'):
            self.validate()

    def test_lapacke_requires_both_routes_and_exact_scope(self):
        self.add_lapacke()
        data = copy.deepcopy(self.data)
        data['witnesses'] = [w for w in data['witnesses'] if not (
            w['symbol']=='LAPACKE_dgetri' and w['bits']==64 and w['route']=='dll')]
        with self.assertRaisesRegex(ValueError, 'Missing completed LAPACKE ABI/route'):
            self.validate(data)
        self.data['lapacke_target_symbols'].pop()
        with self.assertRaisesRegex(ValueError, 'LAPACKE execution scope differs'):
            self.validate()

    def test_lapacke_svd_real_and_integer_workspace_types(self):
        self.add_lapacke()
        self.validate()
        for symbol, replacement in (
            ('LAPACKE_zgesvd', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_zgesvd_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_zgesdd_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
            ('LAPACKE_dgesdd_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Float64}')),
        ):
            data=copy.deepcopy(self.data)
            index=next(i for i,w in enumerate(data['witnesses']) if w['symbol']==symbol and w['bits']==64)
            data['witnesses'][index]['signature']=replacement(data['witnesses'][index]['signature'])
            with self.assertRaisesRegex(ValueError, 'LAPACKE layout/flag/integer/return'):
                self.validate(data)

    def test_lapacke_rejects_unreviewed_registered_signature_contract(self):
        self.add_lapacke()
        for row in self.rows:
            if row['symbol']=='LAPACKE_zpotrs':
                row['family']='new_family_without_review'
        self.data['inventory_registration_sha256']=registration_digest(self.rows)
        with self.assertRaisesRegex(ValueError, 'LAPACKE signature contract missing'):
            self.validate()

class RepositoryExecutionEvidenceTests(unittest.TestCase):
    def test_committed_inventory_matches_verified_execution_snapshot(self):
        import csv
        import json
        import re
        from lbt_execution import load_execution, lapack_rows, cblas_rows, lapacke_rows
        root = Path(__file__).resolve().parents[2]
        with (root / 'docs/src/assets/lbt-symbol-inventory.tsv').open() as inventory:
            rows = list(csv.DictReader(inventory, delimiter='\t'))
        snapshot = root / 'docs/src/assets/lapack-abi-execution.json'
        self.assertTrue(snapshot.is_file(), 'The execution snapshot must be committed')
        native = re.search(r'^revision\s*=\s*"([0-9a-f]+)"', (root / 'deps/upstream.toml').read_text(), re.MULTILINE).group(1)
        executed, data = load_execution(root, rows, native)
        self.assertEqual(len(lapack_rows(rows)), 1106)
        self.assertEqual(len(cblas_rows(rows)), 172)
        self.assertEqual(len(lapacke_rows(rows)), 116)
        self.assertEqual(len(executed), 1278 + len(lapacke_rows(rows)))
        catalogued = {row['symbol'] for row in rows if row['execution_evidence_status'] == 'direct_call_witness'}
        self.assertEqual(catalogued, set(executed))
        for row in rows:
            if row['symbol'] in executed:
                self.assertEqual(row['execution_evidence'], execution_fields(row['symbol'], executed, data)['execution_evidence'])


if __name__ == '__main__':
    unittest.main()
