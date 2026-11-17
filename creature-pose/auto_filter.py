"""자동 거르기 기준 (볼 것 300장으로 고름, 고른 과정은 REPORT_QUAD_PSEUDO.md).
볼 것에서 정밀도가 가장 높으면서 40장 넘게 남는 조합:
  ±5도 돌려도 관절의 80% 이상이 5% 안에 남음 (두 방향 다)
  + 몸 덩어리 solidity(넓이/볼록 껍질) 0.8 이하 (다리 사이가 비어 있음)
  + 채운 그림이 아님 (채운 그림은 규칙이 가장 자주 틀림)"""


def keep(r: dict) -> bool:
    return (min(r["st_rot+5"], r["st_rot-5"]) >= 0.8 and r["solidity"] <= 0.8 and r["style"] != "fill")
