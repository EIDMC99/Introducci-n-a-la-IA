import pandas as pd
import numpy as np

def analizar_etiquetas(df, categoria_col, etiquetas_col):
    df = df.copy()

    # 1. Separar etiquetas por coma y explotar (una etiqueta por fila)
    df[etiquetas_col] = df[etiquetas_col].str.split(",")
    df_exp = df.explode(etiquetas_col)

    # 2. Limpiar espacios en blanco
    df_exp[etiquetas_col] = df_exp[etiquetas_col].str.strip()

    # 3. Frecuencia global de cada etiqueta (mayor a menor)
    frecuencia_global = (
        df_exp[etiquetas_col]
        .value_counts()
        .sort_values(ascending=False)
    )
    frecuencia_global.name = None

    # 4. Resumen por categoría
    total  = df_exp.groupby(categoria_col)[etiquetas_col].count().rename("total_etiquetas")
    unicas = df_exp.groupby(categoria_col)[etiquetas_col].nunique().rename("etiquetas_unicas")

    def obtener_top(grupo):
        return grupo.value_counts().index[0]

    top = df_exp.groupby(categoria_col)[etiquetas_col].apply(obtener_top).rename("etiqueta_top")

    resumen = pd.concat([total, unicas, top], axis=1).reset_index()
    resumen = resumen.rename(columns={categoria_col: "categoria"})
    resumen = resumen.sort_values("categoria").reset_index(drop=True)
    resumen = resumen[["categoria", "total_etiquetas", "etiquetas_unicas", "etiqueta_top"]]

    return (frecuencia_global, resumen)


# --- (Aquí ya tienes pegado el generador generar_caso_de_uso_analizar_etiquetas) ---

if __name__ == "__main__":
    # 1. Obtener datos de prueba
    entrada, salida_esperada = generar_caso_de_uso_analizar_etiquetas()
    freq_esp, resumen_esp = salida_esperada

    # 2. Llamar a la función
    freq_res, resumen_res = analizar_etiquetas(
        entrada["df"],
        entrada["categoria_col"],
        entrada["etiquetas_col"]
    )

    # 3. Mostrar resultados
    print(f"categoria_col : {entrada['categoria_col']}")
    print(f"etiquetas_col : {entrada['etiquetas_col']}")

    print("\n--- Frecuencia global (top 10) ---")
    print(freq_res.head(10))

    print("\n--- Resumen por categoría ---")
    print(resumen_res)

    # 4. Verificación
    freq_ok    = freq_res.equals(freq_esp)
    resumen_ok = resumen_res.equals(resumen_esp)

    print(f"\n¿Frecuencia global correcta? : {'✅ SÍ' if freq_ok else '❌ NO'}")
    print(f"¿Resumen por categoría correcto?: {'✅ SÍ' if resumen_ok else '❌ NO'}")
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if freq_ok and resumen_ok else '❌ NO '}")
