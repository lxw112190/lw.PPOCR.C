import unittest

from tools.package_winforms_demo import resolve_package_version


class WinFormsReleaseVersionTest(unittest.TestCase):
    def test_local_stable_and_prerelease_names(self):
        for value, expected in ((None, "1.2.1"), ("1.2.1", "1.2.1"),
                                ("v1.2.1", "1.2.1"),
                                ("v1.2.1-preview.1", "1.2.1-preview.1")):
            with self.subTest(value=value):
                self.assertEqual(resolve_package_version("1.2.1", value), expected)

    def test_wrong_version_or_unsafe_tag_is_rejected(self):
        for value in ("", "v1.2.0", "vv1.2.1", "v1.2.1/../../bad", " 1.2.1",
                      "1.2.1\n", "1.2.1-", "1.2.1+metadata", "v1.3.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_package_version("1.2.1", value)


if __name__ == "__main__":
    unittest.main()
