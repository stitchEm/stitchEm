import unittest
from utils.ptv import PTV

TEST_PTV_FILEPATH = "./test_data/procedural-4K-nvenc.ptv"


class TestPTV(unittest.TestCase):
    def test(self):
        ptv = PTV.from_file(TEST_PTV_FILEPATH)

        self.assertTrue(ptv["exposure"]["auto"] == False)

        ptv.merge({"exposure": {"new_value": True}})

        self.assertTrue(ptv["exposure"]["new_value"] == True)
        self.assertTrue(ptv["exposure"]["auto"] == False)

        config = ptv.to_config()
        self.assertTrue(config.has("exposure").has("auto").getBool() == False)

        ptv2 = PTV({"exposure": {"another_value": 10}})
        ptv.merge(ptv2)
        self.assertTrue(ptv["exposure"]["another_value"] == 10)

    def test_merge_array(self):
        ptv3 = PTV({"array": [1, 2, {"value": 3, "unchanged_value": 4}]})
        ptv4 = PTV({"array": [5, 6, {"value": 7, "other_value": 8}, 9, {"value": 10}]})
        ptv3.merge(ptv4)
        self.assertDictEqual({"array": [5, 6, {"value": 7, "unchanged_value": 4, "other_value": 8}, 9, {"value": 10}]},
                             ptv3.data)

    def test_merge_dict(self):
        ptv5 = PTV({"a": 1, "b": 2, "c": {"d": 3}, "e": [4, 5]})
        ptv6 = PTV({"b": 6, "c": {"d": 7}, "e": [8, 9, 10], "f": 11})
        ptv5.merge(ptv6)
        self.assertDictEqual({"a": 1, "b": 6, "c": {"d": 7}, "e": [8, 9, 10], "f": 11}, ptv5.data)

    def test_merge_subset(self):
        ptv7 = PTV({"a": 1, "b": {"c": 3}})
        ptv8 = PTV({"d": {"b": {"c": 4, "e": 5}}})
        ptv7.merge(ptv8["d"])
        self.assertDictEqual({"a": 1, "b": {"c": 4, "e": 5}}, ptv7.data)

    def test_filter(self):
        ptv = PTV({"a": 1, "b": {"cc": 3, "dd": 4}, "d": "hello", "e": [{"aa": 1, "bb": 2}, 2, 3, 4]})
        filter = {"a": True, "b": {"cc": True}, "e": [{"bb": True}, True]}
        ptv.filter(ptv, filter)
        self.assertDictEqual({'a': 1, 'b': {'cc': 3}, 'e': [{'bb': 2}, 2]}, ptv.data)
