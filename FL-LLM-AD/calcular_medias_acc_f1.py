import pandas as pd

# Colunas corretas
COL_ACC = "top1"
COL_F1  = "f1"

experimentos = {
    "Poucos_passos_135M": (
        "results_accs_experimentos_poucos_passos_135m_complete.csv",
        "results_f1_experimentos_poucos_passos_135m_complete.csv",
    ),
    "Poucos_passos_360M": (
        "results_accs_experimentos_poucos_passos_360m_complete.csv",
        "results_f1_experimentos_poucos_passos_360m_complete.csv",
    ),
    "Muitos_passos_135M": (
        "results_accs_experimentos_muitos_passos_135m_complete.csv",
        "results_f1_experimentos_muitos_passos_135m_complete.csv",
    ),
    "Muitos_passos_360M": (
        "results_accs_experimentos_muitos_passos_360m_complete.csv",
        "results_f1_experimentos_muitos_passos_360m_complete.csv",
    ),
}

resultados = []

for nome_exp, (acc_file, f1_file) in experimentos.items():
    df_acc = pd.read_csv(acc_file)
    df_f1  = pd.read_csv(f1_file)

    media_acc = df_acc[COL_ACC].mean()
    media_f1  = df_f1[COL_F1].mean()

    resultados.append({
        "Experimento": nome_exp,
        "ACC_top1_Media": media_acc,
        "F1_Media": media_f1
    })

df_resultados = pd.DataFrame(resultados)

df_media_geral = pd.DataFrame({
    "ACC_top1_Media_Geral": [df_resultados["ACC_top1_Media"].mean()],
    "F1_Media_Geral":       [df_resultados["F1_Media"].mean()]
})

with pd.ExcelWriter("media_acc_f1_experimentos.xlsx", engine="openpyxl") as writer:
    df_resultados.to_excel(writer, sheet_name="Media_por_Experimento", index=False)
    df_media_geral.to_excel(writer, sheet_name="Media_Geral", index=False)

print("Excel gerado corretamente com ACC (top1) e F1.")
