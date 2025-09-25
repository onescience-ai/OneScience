import unittest

from onescience.utils.fcn.YParams import YParams


class TestYParams(unittest.TestCase):
    def setUp(self):
        self.yaml_filename = "tests/unittest/test_config.yaml"
        self.config_name = "test_config"
        self.yparams = YParams(self.yaml_filename, self.config_name)

    def test_initialization(self):
        self.assertIsInstance(self.yparams, YParams)
        self.assertIn("param1", self.yparams.params)
        self.assertEqual(self.yparams.params["param1"], "value1")

    def test_get_item(self):
        self.assertEqual(self.yparams["param1"], "value1")

    def test_set_item(self):
        self.yparams["param2"] = "new_value"
        self.assertEqual(self.yparams["param2"], "new_value")

    def test_contains(self):
        self.assertIn("param1", self.yparams)
        self.assertNotIn("non_existent_param", self.yparams)

    def test_update_params(self):
        new_config = {"param1": "updated_value", "param3": "value3"}
        self.yparams.update_params(new_config)
        self.assertEqual(self.yparams["param1"], "updated_value")
        self.assertEqual(self.yparams["param3"], "value3")

    def test_log(self):
        import logging

        logging.basicConfig(level=logging.INFO)
        self.yparams.log()
        # Check if the log contains expected output
        self.assertIn(
            "Configuration file: tests/unittest/test_config.yaml", caplog.text
        )
        self.assertIn("param1 value1", caplog.text)
        self.assertIn("param2 value2", caplog.text)

    def test_parse_env_vars(self):
        import os

        os.environ["TEST_ENV_VAR"] = "env_value"
        data = {
            "param1": "value1",
            "param2": "${TEST_ENV_VAR}",
            "nested": {"param3": "value3", "param4": "${TEST_ENV_VAR}"},
        }
        parsed_data = self.yparams.parse_env_vars(data)
        self.assertEqual(parsed_data["param2"], "env_value")
        self.assertEqual(parsed_data["nested"]["param4"], "env_value")
        del os.environ["TEST_ENV_VAR"]

    def test_parse_env_vars_with_list(self):
        import os

        os.environ["TEST_ENV_VAR"] = "env_value"
        data = {
            "param1": "value1",
            "param2": ["${TEST_ENV_VAR}", "static_value"],
            "nested": {
                "param3": "value3",
                "param4": ["${TEST_ENV_VAR}", "another_value"],
            },
        }
        parsed_data = self.yparams.parse_env_vars(data)
        self.assertEqual(parsed_data["param2"][0], "env_value")
        self.assertEqual(parsed_data["nested"]["param4"][0], "env_value")
        del os.environ["TEST_ENV_VAR"]

    def test_parse_env_vars_no_env_var(self):
        data = {"param1": "value1", "param2": "static_value"}
        parsed_data = self.yparams.parse_env_vars(data)
        self.assertEqual(parsed_data["param2"], "static_value")
        self.assertEqual(parsed_data["param1"], "value1")

    def test_parse_env_vars_empty_string(self):
        data = ""
        parsed_data = self.yparams.parse_env_vars(data)
        self.assertEqual(parsed_data, "")

    def test_parse_env_vars_none(self):
        data = None
        parsed_data = self.yparams.parse_env_vars(data)
        self.assertIsNone(parsed_data)


if __name__ == "__main__":
    unittest.main()
