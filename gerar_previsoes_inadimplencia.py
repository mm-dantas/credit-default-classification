def gerar_previsoes_inadimplencia():
    import warnings
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    import time
    import numpy as np
    from sklearn.preprocessing import LabelEncoder 
    from sklearn.metrics import mean_squared_error,precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
    from imblearn import under_sampling, over_sampling 
    from imblearn.over_sampling import SMOTE 
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.metrics import r2_score 
    import socket
    import pyodbc

    warnings.filterwarnings("ignore") 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    clf = joblib.load('modelo_treinado.pk')
    print("Pacotes Carregados...")

    conexao = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=MODELOS_PREDITIVOS;"
    "UID=usuario_python;"
    "PWD=123456;"
    )
    df_original = pd.read_sql("SELECT * FROM EXTRACAO_DADOS_SISTEMA", conexao)
    conexao.close()

    df_original.dropna(inplace=True)

    bins = [-100, 120, 180, 240]
    labels = ['Até 120 Meses', '121 até 180 Meses', '181 até 240 Meses']
    df_original['FAIXA_PRAZO_FINANCIAMENTO'] = pd.cut(df_original['PZ_FINANCIAMENTO'], bins=bins, labels=labels)
    df_original['FAIXA_PRAZO_FINANCIAMENTO'].value_counts()

    bins = [-100, 100000, 200000, 300000, 400000, 500000, 750000, 1000000, 9000000000]
    labels = ['Até 100 mil', '101 até 200 mil', '201 até 300 mil', '301 até 400 mil', '401 até 500 mil', 
            '501 até 750 mil', 'De 751 até 1.000.000','Mais de 1.000.000']
    df_original['FAIXA_VALOR_FINANCIADO'] = pd.cut(df_original['VALOR_FINANCIAMENTO'], bins=bins, labels=labels)
    df_original['FAIXA_VALOR_FINANCIADO'].value_counts()

    columns = ['TAXA_AO_ANO', 'CIDADE_CLIENTE', 'ESTADO_CLIENTE','RENDA_MENSAL_CLIENTE', 
            'QT_PC_ATRASO', 'QT_DIAS_PRIM_PC_ATRASO','QT_TOTAL_PC_PAGAS',
            'VL_TOTAL_PC_PAGAS', 'QT_PC_PAGA_EM_DIA','QT_DIAS_MIN_ATRASO',
            'QT_DIAS_MAX_ATRASO', 'QT_DIAS_MEDIA_ATRASO','VALOR_PARCELA',
            'IDADE_DATA_ASSINATURA_CONTRATO', 'FAIXA_VALOR_FINANCIADO',
            'FAIXA_PRAZO_FINANCIAMENTO','INADIMPLENTE_COBRANCA',]
    
    df_new = df_original[columns]

    variaveis_categoricas = []
    for i in df_new.columns[0:16].tolist():
        if df_new.dtypes[i] == 'object' or df_new.dtypes[i] == 'category':                        
            variaveis_categoricas.append(i)
    
    lb = LabelEncoder()

    for var in variaveis_categoricas:
        df_new[var] = lb.fit_transform(df_new[var])
    
    variaveis_explicativas = df_new.iloc[:, 0:15]

    Normalizador = MinMaxScaler()
    dados_normalizados = Normalizador.fit_transform(variaveis_explicativas)

    previsoes = clf.predict(dados_normalizados)
    probabilidades = clf.predict_proba(dados_normalizados)
    df_original['PREVISOES'] = previsoes
    df_original['PROBABILIDADES'] = probabilidades[:, 1]

    columns = ['NUMERO_CONTRATO', 'PREVISOES', 'PROBABILIDADES']
    df_conversao = df_original[columns]

    conexao = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=MODELOS_PREDITIVOS;"
    "UID=usuario_python;"
    "PWD=123456;"
    )

    cursor = conexao.cursor()

    for index,row in df_conversao.iterrows():
        sql = "INSERT INTO RESULTADOS_INTERMEDIARIO (NUMERO_CONTRATO, PREVISOES, PROBABILIDADES) VALUES (?, ?, ?)"
        val = (row['NUMERO_CONTRATO'],row['PREVISOES'],row['PROBABILIDADES'])    
        cursor.execute(sql, val)
        conexao.commit()
    
    print("Previsoes Geradas com Sucesso!")

    cursor.execute('EXEC SP_INPUT_RESULTADOS_MODELO_PREDITIVO')    
    conexao.commit()
    conexao.close()
    print("Tabelas Atualizadas! Processo Concluído com Sucesso")


def main():
    gerar_previsoes_inadimplencia()    


if __name__ == "__main__":
    main()    