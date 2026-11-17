"""빠른 종이색 재기가 서비스 그대로와 같은 관절을 내는지 확인 (종류마다 4장 x 꼴 3)."""
import numpy as np
import qd
import rule

same = diff = 0
worst = 0.0
for cat in qd.CATS:
    for d in qd.load(cat, 4):
        for st in qd.STYLES:
            img, _ = qd.render(d, st)
            rule.use_fast(True); a = rule.run(img)
            rule.use_fast(False); b = rule.run(img)
            if a["found"] == b["found"] and a["ok"] == b["ok"] and (a["joints"] is None or np.allclose(a["joints"], b["joints"])):
                same += 1
            else:
                diff += 1
                if a["joints"] is not None and b["joints"] is not None:
                    worst = max(worst, float(np.abs(a["joints"] - b["joints"]).max()))
print(f"같음 {same} / 다름 {diff} / 다를 때 가장 큰 차이 {worst:.1f}px")
