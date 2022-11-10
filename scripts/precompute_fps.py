# Temporarily saving these here
dst_smi = "gs://experiments-output/mood-v2/downstream_applications/virtual_screening.csv"
dst_fps = "gs://experiments-output/mood-v2/downstream_applications/representations/virtual_screening/ECFP.parquet"

path = "gs://screening-libraries/valence-screening/valscreen-v2-2022-03-21.parquet/"
smiles = pd.read_parquet(path, columns=["smiles"])["smiles"].values

rng = np.random.default_rng(42)
smiles = rng.choice(smiles, 50000)
smiles = dm.utils.parallelized(lambda smi: dm.to_smiles(dm.to_mol(smi)), smiles, progress=True)
smiles = [smi for smi in smiles if smi is not None]

df = pd.DataFrame({"canonical_smiles": smiles})
df["inchi_key"] = df["canonical_smiles"].apply(lambda smi: dm.to_inchikey(dm.to_mol(smi)))
df["unique_id"] = df["canonical_smiles"].apply(lambda smi: dm.unique_id(dm.to_mol(smi)))
df = df.drop_duplicates("unique_id")
df = df.dropna()
df.to_csv(dst_smi, index=False)

df["smiles"] = dm.utils.parallelized(standardize_smiles, df["canonical_smiles"].values, n_jobs=-1, progress=True)
df["representation"] = [dm.to_fp(smi) for smi in df["smiles"].values]
df[["unique_id", "representation"]].to_parquet(dst_fps)



import fsspec
dst_smi = "gs://experiments-output/mood-v2/downstream_applications/optimization.csv"
dst_fps = "gs://experiments-output/mood-v2/downstream_applications/representations/optimization/ECFP.parquet"

with fsspec.open("gs://experiments-output/mood/data/reinvent_samples/sampled.smi") as fd:
    smiles = fd.read().decode("utf-8").split("\n")
smiles = dm.utils.parallelized(lambda smi: dm.to_smiles(dm.to_mol(smi)), smiles, progress=True)
smiles = [smi for smi in smiles if smi is not None]

df = pd.DataFrame({"canonical_smiles": smiles})
df["inchi_key"] = df["canonical_smiles"].apply(lambda smi: dm.to_inchikey(dm.to_mol(smi)))
df["unique_id"] = df["canonical_smiles"].apply(lambda smi: dm.unique_id(dm.to_mol(smi)))
df = df.drop_duplicates("unique_id")
df = df.dropna()
df.to_csv(dst_smi, index=False)

df["smiles"] = dm.utils.parallelized(standardize_smiles, df["canonical_smiles"].values, n_jobs=-1, progress=True)
df["representation"] = [dm.to_fp(smi) for smi in df["smiles"].values]
df[["unique_id", "representation"]].to_parquet(dst_fps)