"""Metric tests against hand-computed values.

These are worth more than they look. Every number in the published results table
flows through this file, and a silently wrong nDCG would make the whole project
confidently wrong rather than merely unfinished. Each expected value below is
derived by hand in the comment above it, not captured from a previous run.
"""

import math

import pytest

from rag_evals import metrics


class TestRecall:
    def test_full_and_partial(self):
        ranking = ["a", "b", "c", "d"]
        # one of two gold docs present at k=1 -> 0.5
        assert metrics.recall_at_k(ranking, {"a", "z"}, 1) == 0.5
        assert metrics.recall_at_k(ranking, {"a", "b"}, 2) == 1.0
        assert metrics.recall_at_k(ranking, {"z"}, 10) == 0.0

    def test_chunk_budget_semantics(self):
        """k counts chunks, so repeated docs consume budget.

        Four chunks, three from the same document. At k=3 only 'a' has been
        seen, so recall against {a, b} is 0.5 even though 'b' sits at chunk 4.
        """
        chunks = ["a", "a", "a", "b"]
        assert metrics.recall_at_k(chunks, {"a", "b"}, 3) == 0.5
        assert metrics.recall_at_k(chunks, {"a", "b"}, 4) == 1.0

    def test_negative_question_is_nan(self):
        assert math.isnan(metrics.recall_at_k(["a"], set(), 5))


class TestReciprocalRank:
    def test_first_relevant_position(self):
        # dedup to [a, b, c]; 'c' is third -> 1/3
        assert metrics.reciprocal_rank(["a", "a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert metrics.reciprocal_rank(["a", "b"], {"z"}) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert metrics.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)

    def test_hand_computed(self):
        # ranking [x, a, y, b], gold {a, b}
        # DCG  = 1/log2(3) + 1/log2(5) = 0.630930 + 0.430677 = 1.061607
        # IDCG = 1/log2(2) + 1/log2(3) = 1.000000 + 0.630930 = 1.630930
        # nDCG = 1.061607 / 1.630930 = 0.650921
        got = metrics.ndcg_at_k(["x", "a", "y", "b"], {"a", "b"}, 10)
        assert got == pytest.approx(0.650921, abs=1e-5)

    def test_idcg_respects_k(self):
        """With 3 gold docs but k=2, the ideal ranking can only hold 2."""
        got = metrics.ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 2)
        assert got == pytest.approx(1.0)


class TestAggregation:
    def test_negatives_excluded_not_zeroed(self):
        scored = metrics.score_retrieval(
            [
                (["a"], ["a"]),      # perfect
                (["z"], ["b"]),      # miss
                (["q"], []),         # negative case
            ]
        )
        assert scored.n_scored == 2
        assert scored.n_skipped_negative == 1
        # mean of 1.0 and 0.0, not of 1.0, 0.0 and 0.0
        assert scored.mrr == pytest.approx(0.5)


class TestGenerationScores:
    def test_abstain_on_everything_is_penalised(self):
        """A system that always abstains must not score well.

        Ten questions, two of them unanswerable. Abstaining on all ten gets the
        two negatives right and the eight answerable ones wrong: 2/10 = 0.2.
        """
        records = [
            {"is_negative": i >= 8, "abstained": True, "grounded": True, "correct": False}
            for i in range(10)
        ]
        s = metrics.score_generation(records)
        assert s.abstention_accuracy == pytest.approx(0.2)
        assert s.false_abstentions == 8

    def test_confabulation_counted(self):
        records = [
            {"is_negative": True, "abstained": False, "grounded": False, "correct": False},
            {"is_negative": True, "abstained": True, "grounded": True, "correct": False},
        ]
        s = metrics.score_generation(records)
        assert s.confabulations == 1
        assert s.abstention_accuracy == pytest.approx(0.5)

    def test_groundedness_ignores_abstentions(self):
        """An abstention asserts nothing, so it cannot be ungrounded."""
        records = [
            {"is_negative": False, "abstained": True, "grounded": True, "correct": False},
            {"is_negative": False, "abstained": False, "grounded": False, "correct": True},
        ]
        s = metrics.score_generation(records)
        assert s.groundedness == pytest.approx(0.0)  # only the answered one counts


class TestCohenKappa:
    def test_perfect_agreement(self):
        a = metrics.cohen_kappa([True, False, True], [True, False, True])
        assert a.raw_agreement == 1.0
        assert a.cohen_kappa == 1.0

    def test_degenerate_judge_scores_zero(self):
        """The whole reason kappa is reported instead of raw agreement.

        A judge that answers True unconditionally against a 90% True label set
        gets 90% raw agreement while carrying no information. Kappa is 0.
        """
        human = [True] * 9 + [False]
        judge = [True] * 10
        a = metrics.cohen_kappa(judge, human)
        assert a.raw_agreement == pytest.approx(0.9)
        assert a.cohen_kappa == pytest.approx(0.0)

    def test_disagreement_direction_is_tracked(self):
        a = metrics.cohen_kappa([True, False], [False, False])
        assert a.judge_yes_human_no == 1
        assert a.judge_no_human_yes == 0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            metrics.cohen_kappa([True], [True, False])
