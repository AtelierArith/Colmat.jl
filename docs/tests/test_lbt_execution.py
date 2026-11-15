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


if __name__ == '__main__':
    unittest.main()
