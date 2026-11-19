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
            'LAPACKE_dormqr': ('ormqr', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormqr_work': ('ormqr', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zunmqr': ('unmqr', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmqr_work': ('unmqr', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dormlq': ('ormlq', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormlq_work': ('ormlq', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zunmlq': ('unmlq', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmlq_work': ('unmlq', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dormql': ('ormql', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormql_work': ('ormql', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zunmql': ('unmql', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmql_work': ('unmql', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dormrq': ('ormrq', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormrq_work': ('ormrq', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zunmrq': ('unmrq', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmrq_work': ('unmrq', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dorgqr': ('orgqr', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dorgqr_work': ('orgqr', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zungqr': ('ungqr', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zungqr_work': ('ungqr', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dorglq': ('orglq', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dorglq_work': ('orglq', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zunglq': ('unglq', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zunglq_work': ('unglq', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dorgql': ('orgql', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dorgql_work': ('orgql', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zungql': ('ungql', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zungql_work': ('ungql', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dorgrq': ('orgrq', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dorgrq_work': ('orgrq', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zungrq': ('ungrq', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zungrq_work': ('ungrq', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgeqlf': ('geqlf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgerqf': ('gerqf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgeqlf_work': ('geqlf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgerqf_work': ('gerqf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtpqrt': ('tpqrt', 'd', lambda j: f'(Int32, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtpqrt_work': ('tpqrt', 'd', lambda j: f'(Int32, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dtpqrt2': ('tpqrt2', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtpqrt2_work': ('tpqrt2', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtpmqrt': ('tpmqrt', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtpmqrt_work': ('tpmqrt', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dtprfb': ('tprfb', 'd', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtprfb_work': ('tprfb', 'd', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_ztpqrt': ('tpqrt', 'z', lambda j: f'(Int32, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztpqrt_work': ('tpqrt', 'z', lambda j: f'(Int32, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_ztpqrt2': ('tpqrt2', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztpqrt2_work': ('tpqrt2', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztpmqrt': ('tpmqrt', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztpmqrt_work': ('tpmqrt', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_ztprfb': ('tprfb', 'z', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztprfb_work': ('tprfb', 'z', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_sgels': ('gels', 's', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sgels_work': ('gels', 's', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sgelsy': ('gelsy', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{{j}}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_sgelsy_work': ('gelsy', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{{j}}}, Float32, Ptr{{{j}}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sgelss': ('gelss', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_sgelss_work': ('gelss', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sgelsd': ('gelsd', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_sgelsd_work': ('gelsd', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}}, Ptr{{Float32}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_dgels': ('gels', 'd', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgels_work': ('gels', 'd', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgelsy': ('gelsy', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_dgelsy_work': ('gelsy', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgelss': ('gelss', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_dgelss_work': ('gelss', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgelsd': ('gelsd', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_dgelsd_work': ('gelsd', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_cgels': ('gels', 'c', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_cgels_work': ('gels', 'c', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_cgelsy': ('gelsy', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{{j}}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_cgelsy_work': ('gelsy', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{{j}}}, Float32, Ptr{{{j}}}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}})'),
            'LAPACKE_cgelss': ('gelss', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_cgelss_work': ('gelss', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}})'),
            'LAPACKE_cgelsd': ('gelsd', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}})'),
            'LAPACKE_cgelsd_work': ('gelsd', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}}, Float32, Ptr{{{j}}}, Ptr{{ComplexF32}}, {j}, Ptr{{Float32}}, Ptr{{{j}}})'),
            'LAPACKE_zgels': ('gels', 'z', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgels_work': ('gels', 'z', lambda j: f'(Int32, Int8, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgelsy': ('gelsy', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_zgelsy_work': ('gelsy', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Float64, Ptr{{{j}}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zgelss': ('gelss', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_zgelss_work': ('gelss', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zgelsd': ('gelsd', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}})'),
            'LAPACKE_zgelsd_work': ('gelsd', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Float64, Ptr{{{j}}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{{j}}})'),
            'LAPACKE_sggqrf': ('ggqrf', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j}, Ptr{{Float32}})'),
            'LAPACKE_sggqrf_work': ('ggqrf', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sggrqf': ('ggrqf', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j}, Ptr{{Float32}})'),
            'LAPACKE_sggrqf_work': ('ggrqf', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sgglse': ('gglse', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}})'),
            'LAPACKE_sgglse_work': ('gglse', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_sggglm': ('ggglm', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}})'),
            'LAPACKE_sggglm_work': ('ggglm', 's', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, {j}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}}, Ptr{{Float32}}, {j})'),
            'LAPACKE_dggqrf': ('ggqrf', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dggqrf_work': ('ggqrf', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dggrqf': ('ggrqf', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dggrqf_work': ('ggrqf', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgglse': ('gglse', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}})'),
            'LAPACKE_dgglse_work': ('gglse', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dggglm': ('ggglm', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}})'),
            'LAPACKE_dggglm_work': ('ggglm', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_cggqrf': ('ggqrf', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}})'),
            'LAPACKE_cggqrf_work': ('ggqrf', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_cggrqf': ('ggrqf', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}})'),
            'LAPACKE_cggrqf_work': ('ggrqf', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_cgglse': ('gglse', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}})'),
            'LAPACKE_cgglse_work': ('gglse', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_cggglm': ('ggglm', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}})'),
            'LAPACKE_cggglm_work': ('ggglm', 'c', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, {j}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, Ptr{{ComplexF32}}, {j})'),
            'LAPACKE_zggqrf': ('ggqrf', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zggqrf_work': ('ggqrf', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zggrqf': ('ggrqf', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zggrqf_work': ('ggrqf', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgglse': ('gglse', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgglse_work': ('gglse', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zggglm': ('ggglm', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}})'),
            'LAPACKE_zggglm_work': ('ggglm', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dtzrzf': ('tzrzf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dtzrzf_work': ('tzrzf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormrz': ('ormrz', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dormrz_work': ('ormrz', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_ztzrzf': ('tzrzf', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_ztzrzf_work': ('tzrzf', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmrz': ('unmrz', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zunmrz_work': ('unmrz', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgeqp3': ('geqp3', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}})'),
            'LAPACKE_dgeqp3_work': ('geqp3', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeqpf': ('geqpf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}})'),
            'LAPACKE_dgeqpf_work': ('geqpf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}})'),
            'LAPACKE_zgeqp3': ('geqp3', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgeqp3_work': ('geqp3', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zgeqpf': ('geqpf', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgeqpf_work': ('geqpf', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, Ptr{{Float64}})'),
            'LAPACKE_dgeqrt': ('geqrt', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeqrt_work': ('geqrt', 'd', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgeqrt2': ('geqrt2', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeqrt2_work': ('geqrt2', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeqrt3': ('geqrt3', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeqrt3_work': ('geqrt3', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgemqrt': ('gemqrt', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgemqrt_work': ('gemqrt', 'd', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zgeqrt': ('geqrt', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgeqrt_work': ('geqrt', 'z', lambda j: f'(Int32, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgeqrt2': ('geqrt2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgeqrt2_work': ('geqrt2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgeqrt3': ('geqrt3', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgeqrt3_work': ('geqrt3', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgemqrt': ('gemqrt', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgemqrt_work': ('gemqrt', 'z', lambda j: f'(Int32, Int8, Int8, {j}, {j}, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_dgelqf': ('gelqf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgelqf_work': ('gelqf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zgelq2': ('gelq2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgelq2_work': ('gelq2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}})'),
            'LAPACKE_dgetri': ('getri', 'd', lambda j: f'(Int32, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_dgetri_work': ('getri', 'd', lambda j: f'(Int32, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zpotrs': ('potrs', 'z', lambda j: f'(Int32, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zsyr': ('syr', 'z', lambda j: f'(Int32, Int8, {j}, ComplexF64, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgeqrfp': ('geqrfp', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgeqrfp_work': ('geqrfp', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zgeqrfp': ('geqrfp', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgeqrfp_work': ('geqrfp', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgeqrf': ('geqrf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgeqrf_work': ('geqrf', 'd', lambda j: f'(Int32, {j}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zgeqr2': ('geqr2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}})'),
            'LAPACKE_zgeqr2_work': ('geqr2', 'z', lambda j: f'(Int32, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}})'),
            'LAPACKE_dlauum': ('lauum', 'd', lambda j: f'(Int32, Int8, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dlauum_work': ('lauum', 'd', lambda j: f'(Int32, Int8, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zlauum': ('lauum', 'z', lambda j: f'(Int32, Int8, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zlauum_work': ('lauum', 'z', lambda j: f'(Int32, Int8, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dtrtri': ('trtri', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dtrtri_work': ('trtri', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_ztrtrs': ('trtrs', 'z', lambda j: f'(Int32, Int8, Int8, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_ztrtrs_work': ('trtrs', 'z', lambda j: f'(Int32, Int8, Int8, Int8, {j}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_dgeevx': ('geevx', 'd', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}})'),
            'LAPACKE_dgeevx_work': ('geevx', 'd', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_zgeevx': ('geevx', 'z', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}})'),
            'LAPACKE_zgeevx_work': ('geevx', 'z', lambda j: f'(Int32, Int8, Int8, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dgeev': ('geev', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_dgeev_work': ('geev', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, {j})'),
            'LAPACKE_zgeev': ('geev', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j})'),
            'LAPACKE_zgeev_work': ('geev', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
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
            'LAPACKE_dsyevd': ('syevd', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_dsyevd_work': ('syevd', 'd', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, {j})'),
            'LAPACKE_zheevd': ('heevd', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}})'),
            'LAPACKE_zheevd_work': ('heevd', 'z', lambda j: f'(Int32, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, {j})'),
            'LAPACKE_dsyevr': ('syevr', 'd', lambda j: f'(Int32, Int8, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Float64, Float64, {j}, {j}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_dsyevr_work': ('syevr', 'd', lambda j: f'(Int32, Int8, Int8, Int8, {j}, Ptr{{Float64}}, {j}, Float64, Float64, {j}, {j}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, {j})'),
            'LAPACKE_zheevr': ('heevr', 'z', lambda j: f'(Int32, Int8, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Float64, Float64, {j}, {j}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}})'),
            'LAPACKE_zheevr_work': ('heevr', 'z', lambda j: f'(Int32, Int8, Int8, Int8, {j}, Ptr{{ComplexF64}}, {j}, Float64, Float64, {j}, {j}, Float64, Ptr{{{j}}}, Ptr{{Float64}}, Ptr{{ComplexF64}}, {j}, Ptr{{{j}}}, Ptr{{ComplexF64}}, {j}, Ptr{{Float64}}, {j}, Ptr{{{j}}}, {j})'),
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

    def test_lapacke_real_and_integer_workspace_types(self):
        self.add_lapacke()
        self.validate()
        for symbol, replacement in (
            ('LAPACKE_dgglse_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zggqrf', lambda s: s.replace('Ptr{ComplexF64}', 'Ptr{Float64}', 1)),
            ('LAPACKE_zggglm_work', lambda s: s[:-1] + ', Ptr{Float64})'),
            ('LAPACKE_dggrqf', lambda s: s.replace('Int32, Int64', 'Int64, Int64', 1)),
            ('LAPACKE_dgels_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zgelsy', lambda s: s.replace('Float64, Ptr{Int64}', 'Ref{Float64}, Ptr{Int64}')),
            ('LAPACKE_zgelsy_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_dgelss_work', lambda s: s[:-1] + ', Ptr{Int64})'),
            ('LAPACKE_dgelsd_work', lambda s: s[:-len('Ptr{Int64})')] + 'Ptr{Int32})'),
            ('LAPACKE_zgelsd_work', lambda s: s.replace(', Ptr{Float64}, Ptr{Int64})', ', Ptr{ComplexF64}, Ptr{Int64})')),
            ('LAPACKE_dormrz_work', lambda s: s.replace('Int64, Int64, Int64, Int64', 'Int64, Int64, Int64, Int32', 1)),
            ('LAPACKE_zunmrz', lambda s: s.replace('Int8, Int8', 'Int8, Int32', 1)),
            ('LAPACKE_ztzrzf_work', lambda s: s.replace('Ptr{ComplexF64}', 'Ptr{Float64}')),
            ('LAPACKE_dgeqp3', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
            ('LAPACKE_zgeqp3_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_dgeqpf_work', lambda s: s[:-1] + ', Int64)'),
            ('LAPACKE_dgemqrt_work', lambda s: s.replace('Int8, Int8', 'Int8, Int32', 1)),
            ('LAPACKE_zgeqrt_work', lambda s: s.replace('Ptr{ComplexF64}', 'Ptr{Float64}')),
            ('LAPACKE_dgeqrt2_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_dgeqrfp_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zgeqrfp', lambda s: s[:-len('Ptr{ComplexF64})')] + 'Ptr{Float64})'),
            ('LAPACKE_dgeqrf_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_dormqr_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zunmlq', lambda s: s.replace('Int8, Int8', 'Int8, Int32', 1)),
            ('LAPACKE_dorgqr_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zunglq', lambda s: s.replace('Int64, Int64, Int64', 'Int64, Int64, Int32', 1)),
            ('LAPACKE_dgeqlf_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_dgerqf_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_dgelqf_work', lambda s: s[:-len('Int64)')] + 'Int32)'),
            ('LAPACKE_zgelq2', lambda s: s[:-len('Ptr{ComplexF64})')] + 'Ptr{Float64})'),
            ('LAPACKE_zgeqr2', lambda s: s[:-len('Ptr{ComplexF64})')] + 'Ptr{Float64})'),
            ('LAPACKE_dlauum', lambda s: s.replace('Int64','Int32')),
            ('LAPACKE_zlauum_work', lambda s: s.replace('Ptr{ComplexF64}','Ptr{Float64}')),
            ('LAPACKE_dtrtri_work', lambda s: s.replace('Int64', 'Int32')),
            ('LAPACKE_ztrtrs_work', lambda s: s.replace('Ptr{ComplexF64}', 'Ptr{Float64}')),
            ('LAPACKE_dgeevx_work', lambda s: s[:-len('Ptr{Int64})')] + 'Ptr{Float64})'),
            ('LAPACKE_zgeevx_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{Int64})'),
            ('LAPACKE_zgeevx', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
            ('LAPACKE_zgeev_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_dgeev', lambda s: s.replace('Ptr{Float64}, Ptr{Float64}, Ptr{Float64}', 'Ptr{Float64}, Ptr{ComplexF64}, Ptr{Float64}')),
            ('LAPACKE_zgeev', lambda s: s.replace('Int64', 'Int32')),
            ('LAPACKE_zgesvd', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_zgesvd_work', lambda s: s[:-len('Ptr{Float64})')] + 'Ptr{ComplexF64})'),
            ('LAPACKE_zgesdd_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
            ('LAPACKE_dgesdd_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Float64}')),
            ('LAPACKE_zheevd_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
            ('LAPACKE_zheevd_work', lambda s: s.replace(', Ptr{Float64}, Int64, Ptr{Int64}', ', Ptr{ComplexF64}, Int64, Ptr{Int64}')),
            ('LAPACKE_zheevr', lambda s: s.replace(', Float64, Float64,', ', ComplexF64, Float64,')),
            ('LAPACKE_zheevr_work', lambda s: s.replace('Ptr{Int64}', 'Ptr{Int32}')),
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
        self.assertEqual(len(lapack_rows(rows)), 1120)
        self.assertEqual(len(cblas_rows(rows)), 172)
        self.assertEqual(len(lapacke_rows(rows)), 1540)
        self.assertEqual(len(executed), 1292 + len(lapacke_rows(rows)))
        catalogued = {row['symbol'] for row in rows if row['execution_evidence_status'] == 'direct_call_witness'}
        self.assertEqual(catalogued, set(executed))
        for row in rows:
            if row['symbol'] in executed:
                self.assertEqual(row['execution_evidence'], execution_fields(row['symbol'], executed, data)['execution_evidence'])


if __name__ == '__main__':
    unittest.main()
