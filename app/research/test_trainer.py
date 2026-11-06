"""Small correctness checks for the controlled TRNet experiment protocol."""
import copy
import math
import unittest

import torch
from torch import nn

from model import TRNet
from model_train.TRNet_train import TRNet_Trainer
from app.research.summarize import aggregate, select_winners


def events(src, dst, timestamp):
    return tuple(torch.tensor(value, dtype=torch.long) for value in (src, dst, timestamp))


def samples(src, dst, timestamp, labels):
    return {
        "src": torch.tensor(src, dtype=torch.long),
        "dst": torch.tensor(dst, dtype=torch.long),
        "query_t": torch.full((len(src),), timestamp, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
    }


class StaticScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scores", torch.tensor([3.0, -2.0, 0.0, 1.0, 3.0]))

    def update_mem(self, src, dst, event_t):
        pass

    def forward(self, src, dst):
        return self.scores[src]


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.train = [events([1], [2], [1]), events([2], [3], [2])]
        self.val = [events([3], [4], [3])]
        self.train_samples = [
            samples([1, 3], [2, 1], 1, [1, 0]),
            samples([1, 4], [3, 1], 2, [1, 0]),
        ]
        self.val_samples = [samples([1, 4], [4, 1], 3, [1, 0])]

    def model(self, encoder="residual_gru"):
        model = TRNet(mem_dim=8, latent_dim=8, encoder_type=encoder)
        model.set_module(n_node=4)
        return model

    def test_validation_rebuilds_prefix_with_current_weights(self):
        model = self.model()
        TRNet_Trainer.replay_memory(model, self.train)
        # Change the encoder after the old prefix was processed; a valid
        # evaluation must discard this stale graph state and rebuild it.
        with torch.no_grad():
            for parameter in model.encoder.parameters():
                parameter.add_(0.2)
        model.module.cumul_mem.fill_(99)
        actual = TRNet_Trainer.evaluate_research(
            model, self.val, self.val_samples, prefix_batches=(self.train,)
        )
        actual_state = model.module.snapshot_state()
        model.reset_memory()
        TRNet_Trainer.replay_memory(model, self.train, reset=False)
        expected = TRNet_Trainer.evaluate_research(model, self.val, self.val_samples)
        self.assertEqual(actual, expected)
        for name, value in actual_state.items():
            torch.testing.assert_close(value, getattr(model.module, name), rtol=0, atol=0)
        self.assertIsNone(model.module.cumul_mem.grad_fn)

    def test_loss_and_metrics_cover_all_samples_with_unequal_batch_sizes(self):
        batch_events = [events([1], [2], [1]), events([2], [3], [2])]
        batch_samples = [
            samples([0], [1], 1, [1]),
            samples([1, 2, 3, 4], [1, 1, 1, 1], 2, [0, 1, 0, 1]),
        ]
        result = TRNet_Trainer.evaluate_research(StaticScores(), batch_events, batch_samples)
        expected = nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([3.0, -2.0, 0.0, 1.0, 3.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
        ).item()
        self.assertAlmostEqual(result["loss"], expected, places=7)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["acc"], 0.8)
        self.assertTrue(0 <= result["ap"] <= 1)
        self.assertTrue(0 <= result["auc"] <= 1)

    def test_rejects_misaligned_samples(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            TRNet_Trainer.evaluate_research(self.model(), self.val, [])
        bad_samples = copy.deepcopy(self.val_samples)
        bad_samples[0]["query_t"].fill_(4)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            TRNet_Trainer.evaluate_research(self.model(), self.val, bad_samples)

    def test_two_epoch_training_and_multibatch_backprop_for_all_encoders(self):
        for encoder in ("residual_gru", "normalized_gru", "diagonal", "lru", "gated_linear", "mingru"):
            with self.subTest(encoder=encoder):
                model = self.model(encoder)
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                requested_epochs = []

                def sample_factory(epoch):
                    requested_epochs.append(epoch)
                    return self.train_samples

                result = TRNet_Trainer.fit_research(
                    model, self.train, self.val, self.val_samples, sample_factory,
                    epochs=2, patience=2, bptt_batches=2,
                )
                self.assertEqual(requested_epochs, [0, 1])
                self.assertEqual(result["epochs_run"], 2)
                self.assertIn(result["best_epoch"], [1, 2])
                self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()))
                for row in result["history"]:
                    self.assertTrue(math.isfinite(row["train_loss"]))
                self.assertIsNone(model.module.cumul_mem.grad_fn)
                rebuilt = TRNet_Trainer.evaluate_research(
                    model, self.val, self.val_samples, prefix_batches=(self.train,)
                )
                for metric in ("loss", "acc", "ap", "auc"):
                    self.assertAlmostEqual(rebuilt[metric], result["validation"][metric], places=7)


class SummaryTests(unittest.TestCase):
    def test_selection_uses_validation_even_if_other_config_has_better_test(self):
        records = []
        for encoder, val_auc, test_auc in (("a", 0.9, 0.6), ("b", 0.8, 0.99)):
            for seed in (1, 2, 3):
                records.append({
                    "stage": "confirm", "path": f"{encoder}_{seed}.json",
                    "config": {"encoder_type": encoder, "seed": seed},
                    "validation": {"acc": val_auc, "ap": val_auc, "auc": val_auc},
                    "test": {"acc": test_auc, "ap": test_auc, "auc": test_auc},
                })
        winner = select_winners(aggregate(records), 3)["confirm"]
        self.assertEqual(winner["config"]["encoder_type"], "a")


if __name__ == "__main__":
    unittest.main()
