import glob
import re
import pandas as pd

PLAYERS = ("player1", "player2")

files = sorted(glob.glob("raw/sub-**/eeg/*.tsv", recursive=True))

df = pd.concat(
    [
        pd.read_csv(f, sep="\t").assign(session=re.search(r"sub-[^/\\]+", f).group(0))
        for f in files
    ],
    ignore_index=True,
)
df = df.set_index(["session", "trial_num"]).sort_index()

# idgaf just need it into a df object
class Temp:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def dyad(self, actor: str, partner: str) -> pd.DataFrame:
        self_resp = self.df[f"{actor}_resp"].mask(self.df[f"{actor}_resp"] == 0)
        partner_resp = self.df[f"{partner}_resp"].mask(self.df[f"{partner}_resp"] == 0)
        self_rt = self.df[f"{actor}_rt"].mask(self.df[f"{actor}_rt"] == 100)
        partner_rt = self.df[f"{partner}_rt"].mask(self.df[f"{partner}_rt"] == 100)
        return pd.DataFrame({
            "self_resp": self_resp,
            "self_rt": self_rt,
            "partner_resp": partner_resp,
            "partner_rt": partner_rt,
        })


sessions = Temp(df)
dyads_df = pd.concat(
    [sessions.dyad(actor, partner) for actor, partner in zip(PLAYERS, reversed(PLAYERS))],
    keys=PLAYERS,
    names=["player", *df.index.names],
).sort_index()

dyads_df.to_excel("data/output.xlsx", index=True)
