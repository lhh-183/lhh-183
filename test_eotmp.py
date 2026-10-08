import unittest

from eotmp import (
    BloomFilter,
    bloom_fpp_estimate,
    choose_bloom_parameters,
    make_synthetic_sets,
    murmur3_32,
    polynomial_coefficients,
    run_eotmp,
)


def eval_polynomial(coefficients, value):
    return sum(coefficient * value**power for power, coefficient in enumerate(coefficients))


class EoTMPTests(unittest.TestCase):
    def test_murmur_and_bloom_are_deterministic(self):
        self.assertEqual(murmur3_32(b"abc"), murmur3_32(b"abc"))
        first = BloomFilter.from_set({"a", "b", "c"}, 128, 5, 42)
        second = BloomFilter.from_set({"a", "b", "c"}, 128, 5, 42)
        self.assertEqual(first.values, second.values)
        self.assertTrue(all(first.contains(item) for item in ("a", "b", "c")))

    def test_threshold_polynomial_has_expected_roots(self):
        coefficients = polynomial_coefficients([4, 5, 6])
        self.assertEqual([eval_polynomial(coefficients, x) for x in (4, 5, 6)], [0, 0, 0])
        self.assertNotEqual(eval_polynomial(coefficients, 3), 0)

    def test_protocol_has_no_false_negatives(self):
        sets = make_synthetic_sets(8, 64, 6, seed=11)
        result = run_eotmp(sets, 6, bloom_bits=16384, bloom_hashes=10, rng_seed=11)
        self.assertEqual(result.false_negatives, set())
        self.assertTrue(result.exact.issubset(result.result))

    def test_complement_polynomial_path_has_no_false_negatives(self):
        sets = make_synthetic_sets(8, 64, 2, seed=12)
        result = run_eotmp(sets, 2, bloom_bits=16384, bloom_hashes=10, rng_seed=12)
        self.assertEqual(result.stats.polynomial_degree, 2)
        self.assertEqual(result.false_negatives, set())
        self.assertTrue(result.exact.issubset(result.result))

    def test_online_subset_is_supported(self):
        sets = make_synthetic_sets(5, 24, 3, seed=4)
        result = run_eotmp(sets, 2, online=[0, 1, 2], receiver=0, bloom_bits=4096, bloom_hashes=8)
        expected = {item for item in sets[0] if sum(item in sets[i] for i in (0, 1, 2)) >= 2}
        self.assertEqual(result.exact, expected)
        self.assertEqual(result.false_negatives, set())

    def test_parameter_search_meets_target(self):
        bits, hashes, fpp = choose_bloom_parameters(64, 1e-3)
        self.assertGreaterEqual(bits, 64)
        self.assertGreaterEqual(hashes, 1)
        self.assertLessEqual(fpp, 1e-3)
        self.assertAlmostEqual(fpp, bloom_fpp_estimate(bits, hashes, 64))


if __name__ == "__main__":
    unittest.main()
