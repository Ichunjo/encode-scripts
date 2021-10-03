
from vardefunc.noise import AddGrain


graigasm_args = dict(
    thrs=[x << 8 for x in (32, 80, 128, 176)],
    strengths=[(0.4, 0.3), (0.2, 0.1), (0.15, 0.0), (0.0, 0.0)],
    sizes=(1.2, 1.1, 1, 1),
    sharps=(70, 60, 50, 50),
    grainers=[
        AddGrain(seed=333, constant=False),
        AddGrain(seed=333, constant=False),
        AddGrain(seed=333, constant=True)
    ]
)
