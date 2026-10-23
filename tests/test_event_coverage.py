import unittest
from dreamer_agent.event_coverage import scan, task_for, DREAMER4_TASKS

K = "minecraft.pickup:minecraft.oak_log"
MINE = "minecraft.mine_block:minecraft.oak_log"


def row(t, stats):
    return dict(tick=t, serverTick=t, stats=stats)


class EventTests(unittest.TestCase):
    def test_table4_order(self):
        self.assertEqual(len(DREAMER4_TASKS), 20)
        self.assertEqual(DREAMER4_TASKS[0], "mine_log")
        self.assertEqual(DREAMER4_TASKS[-1], "use_iron_pickaxe")

    def test_task_for_dreamer_names(self):
        self.assertEqual(task_for(MINE), "mine_log")
        self.assertEqual(task_for(K), "mine_log")
        self.assertEqual(task_for("minecraft.craft_item:minecraft.stick"), "craft_stick")
        self.assertEqual(task_for("minecraft.custom:minecraft.interact_with_crafting_table"), "open_crafting_table")
        self.assertEqual(task_for("minecraft.use_item:minecraft.wooden_pickaxe"), "use_wooden_pickaxe")
        self.assertEqual(task_for("minecraft.craft_item:minecraft.iron_ingot"), "craft_iron_ingot")
        self.assertIsNone(task_for("minecraft.custom:minecraft.jump"))

    def test_initial_history_not_event(self):
        self.assertEqual(scan([row(0, {}), row(1, {K: 10}), row(2, {K: 10})])["events"], [])

    def test_increment_not_cumulative(self):
        e = scan([row(0, {K: 10}), row(1, {K: 12})])["events"]
        self.assertEqual([(x["delta"], x["frame"], x["kind"], x["task"]) for x in e],
                         [(2, 1, "existing_counter_increment", "mine_log")])

    def test_new_key_is_candidate(self):
        e = scan([row(0, {"minecraft.custom:minecraft.jump": 1}),
                  row(1, {K: 1, "minecraft.custom:minecraft.jump": 1})])["events"]
        self.assertEqual(e[0]["kind"], "new_key_candidate")
        self.assertEqual(e[0]["task"], "mine_log")

    def test_gap_unknown_not_positive(self):
        self.assertEqual(scan([row(0, {K: 1}), row(1, {}), row(2, {K: 3})])["events"], [])

    def test_tick_gap(self):
        self.assertEqual(scan([row(0, {K: 1}), row(4, {K: 4})])["events"], [])

    def test_counter_reset(self):
        self.assertEqual(scan([row(0, {K: 10}), row(1, {K: 1})])["events"], [])


if __name__ == "__main__":
    unittest.main()
