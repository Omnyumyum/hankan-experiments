"""「몸 전체 옆모습」 자동 고르기 기준 (볼 것 340장으로 정함, results/body_side_labels.csv)."""


def body_side(f: dict) -> bool:
    return (f["legs"] >= 2 and f["legs_med"] >= 2 and f["low_gap"] >= 0.1
            and f["extent"] <= 0.7 and f["solidity"] <= 0.9)
