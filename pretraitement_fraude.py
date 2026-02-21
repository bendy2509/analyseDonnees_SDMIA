# Importation des librairies
import pandas as pd
import numpy as np

# model_selection : le module de sklearn qui permet de découper, tester et optimiser les modèles.
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

#  les transformateurs sont des objets utilisés dans les pipelines pour transformer les données
from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer    # Remplace les valeurs manquantes par une valeur calculée ou définie.
from sklearn.compose import ColumnTransformer  # permet d’appliquer différentes transformations à différentes colonnes
from sklearn.pipeline import Pipeline    # permet de chaîner plusieurs étapes de traitement et de modélisation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOGIQUE DE TRANSFORMATION
# ==========================================

# Fonction pour la conversion des montants
def transform_montant(X, taux=130):
    """Convertit montant en Gdes.
    - Si contient 'US' => valeur USD * 130
    - Sinon si contient k => valeur milier*1000
    Sinon => valeur deja en Gdes
    - Valeurs non convertibles => NaN
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col]

    # Transformer en string, strip + uppercase
    s_str = s.astype(str).str.strip()
    s_up = s_str.str.upper()

    # Detecter USD
    is_usd = s_up.str.contains("US", na=False)

    # Detecter k (miliers)
    is_k = s_up.str.contains("K", na=False)

    # Retirer "US", ',', 'K' et tout ce qui n'est pas un chiffre ou un point
    s_clean = s_up.astype(str).str.replace(r'[^\d\.]', '', regex=True)

    # Convertir en float
    num = pd.to_numeric(s_clean, errors="coerce")

    # Conversion: USD -> Gdes * taux, sinon garder tel quel
    out = num.where(~is_usd, num * taux)

    # Conversion: k -> *1000
    out = out.where(~is_k, out * 1000)

    return pd.DataFrame(out, columns=[col])


def transform_date_features(X, _cols=True):
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    
    # Nettoyage de base
    s_clean = X[col].astype(str).str.strip()

    # Retirer les points et les virgules
    s_clean = s_clean.str.replace(r'[.,]', '', regex=True)

    # liste de formats
    date_formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%b %d %Y',
        '%B %d %Y',
    ]
    
    # Construire une series de pandas avec des NaT
    result = pd.Series(pd.NaT, index=s_clean.index)

    for fmt in date_formats:
        # On ne tente de recuperer les valeurs encore en NaT 
        mask_missing = result.isna()

        # Si il n'y a plus de valeurs manquantres, on peut arreter
        if not mask_missing.any():
            break
            
        convert = pd.to_datetime(s_clean[mask_missing], format=fmt, errors='coerce')
        result[mask_missing] = convert

    if not _cols:
        return result

    # des features pour le modele
    features = pd.DataFrame(index=X.index)
    features['jour_semaine'] = result.dt.weekday
    features['mois'] = result.dt.month
    features['jour_mois'] = result.dt.day
    features['date_invalide'] = result.isna().astype(int)
    
    return features

# Fonction pour le nettoyage de la devise
def transform_device(X):
    """
    Nettoyer la colonne Devise en remplaçant USD par HTG
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    
    s_clean = (
        X[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("USD", "HTG")
    )
    
    return pd.DataFrame(s_clean, columns=[col])


# Fonction pour convertir l'anciennete en jours
def transform_anciennete_en_jours(X):
    """
    Convertir l'anciennete en jours
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col].astype(str).str.strip().str.lower()

    # Detection des unites
    is_years = s.str.contains(r'y|year|an', regex=True, na=False)
    is_months = s.str.contains(r'm|month|months', regex=True, na=False)
    
    # Extraction du chiffre
    num_str = s.str.replace(r'[^\d]', '', regex=True)
    num = pd.to_numeric(num_str, errors='coerce')

    # Application la conversion en jours
    days = num.copy()
    days.loc[is_years] = num.loc[is_years] * 365
    days.loc[is_months] = num.loc[is_months] * 30

    return pd.DataFrame(days, columns=[col])


# Fonction pour la normalisation des villes
def normaliser_ville(X):
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    
    # Nettoyage de base
    s_clean = X[col].astype(str).str.strip().str.lower()

    # Dictionnaire de normalisation
    mapping = {
        'p-au-p': 'Port-au-Prince',
        'port au prince': 'Port-au-Prince',
        'pap': 'Port-au-Prince',
        'port-au-prince': 'Port-au-Prince',
        'jacmel': 'Jacmel',
        'cap': 'Cap-Haïtien',
        'cap haitien': 'Cap-Haïtien',
        'cap-haïtien': 'Cap-Haïtien',
        'hin': 'Hinche',
        'hinche': 'Hinche',
        'gonaives': 'Gonaïves',
        'gonaïves': 'Gonaïves',
        'cayes': 'Les Cayes',
        'les cayes': 'Les Cayes'
    }

    # Application du remplacement
    s_clean = s_clean.replace(mapping)

    # On remplace les chaines vides ou 'nan' par 'Inconnu'
    s_clean = s_clean.replace(['', 'nan', 'n/a', 'none'], 'Inconnu')

    return pd.DataFrame(s_clean, columns=[col])


# Fonction pour le nettoyage de l'age
def transform_age(X):
    """
    Nettoyer la colonne age en convertissant en numerique 
    et en gerant les valeurs aberrantes age < 18 ou > 90 ans   
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col]

    # Nettoyage de base
    s_clean = s.astype(str).str.strip()
    num = pd.to_numeric(s_clean, errors='coerce')

    # ages entre 18 et 90 ans
    num = num.clip(lower=18, upper=90)

    return pd.DataFrame(num, columns=[col])


# Fonction pour le nettoyage du revenu
def transform_revenu_or_dette(X):
    """
    Nettoyer la colonne RevenuMensuel_raw ou Dette_raw 
    en convertissant en numerique et en gerant les valeurs manquantes    
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col]

    # Nettoyage de base et remplace les virgules
    s_clean = (
        s.astype(str).str.strip()
        .str.replace(',', '', regex=False)
    )

    # Convertir en float
    num = pd.to_numeric(s_clean, errors='coerce')

    return pd.DataFrame(num, columns=[col])


# Fonction pour la normalisation de Employe
def normaliser_employe(X):
    """
    Normaliser la colonne Employe en gerant les valeurs manquantes
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col]

    # Nettoyage de base
    s_clean = s.astype(str).str.strip().str.lower()

    # Dictionnaire de normalisation
    mapping = {
        'yes': 'Oui',
        'no': 'Non',
    }

    # Application du remplacement
    s_clean = s_clean.replace(mapping)

    # On remplace les chaines vides ou 'nan' par 'Inconnu'
    s_clean = s_clean.replace(['', 'nan', 'n/a', 'none'], 'Inconnu')

    # Mettre en majuscule la premiere lettre
    s_clean = s_clean.str.title()

    # Remplace les oui par 1 et les non par 0
    s_clean = s_clean.map({'Oui': 1, 'Non': 0, 'Inconnu': np.nan})

    # Convertir en float
    s_clean_float = pd.to_numeric(s_clean, errors='coerce')

    return pd.DataFrame(s_clean_float, columns=[col])


# Fonction pour traiter les Device
def normaliser_other(X):
    """
    Normaliser la colonne Device en gerant les valeurs manquantes
    """
    X = pd.DataFrame(X).copy()
    col = X.columns[0]
    s = X[col]

    # Nettoyage de base
    s_clean = s.astype(str).str.strip().str.lower()

    # On remplace les chaines vides ou 'nan' par 'Inconnu'
    s_clean = s_clean.replace(['', 'nan', 'n/a', 'none'], 'Inconnu')

    # Mettre en majuscule la premiere lettre
    s_clean = s_clean.str.title()

    return pd.DataFrame(s_clean, columns=[col])

# Fonction pour renommer les colonnes de X avant le pipeline
def rename_columns(X):
    """
        Renommer les colonnes de X pour eviter les confusions entre les brutes et les pretraitees
    """
    X = pd.DataFrame(X).copy()
    _pour_renommer = {
        'Montant_raw': 'Montant_HTG',
        'AncienneteCompte_raw': 'Anciennete_Jours',
        'RevenuMensuel_raw': 'Revenu_Mensuel',
        'DateTransaction_raw': 'DateTransaction',
        'Dette_raw': 'Dette',
        'Age': 'Age_Clean',
        'Employe': 'Employe_Statut',
        'Ville_raw': 'Ville_Clean',
        'Device': 'Device_Clean', 
        'Devise_indiquee': 'Devise',
    }
    X = X.rename(columns=_pour_renommer)
    return X

# Fonction pour exporter les données pretraitées
def export_data_clean(X, y):
    """
        Exporte un fichier Excel avec les donnees pretraitees
    """
    
    # Creer une copie de X pour le nettoyage
    X_clean_excel = X.copy()

    X_clean_excel['Montant_HTG'] = transform_montant(X[['Montant_HTG']])
    X_clean_excel['Anciennete_Jours'] = transform_anciennete_en_jours(X[['Anciennete_Jours']])
    X_clean_excel['Revenu_Mensuel'] = transform_revenu_or_dette(X[['Revenu_Mensuel']])
    X_clean_excel['DateTransaction'] = transform_date_features(X[['DateTransaction']], _cols = False)
    X_clean_excel['Dette'] = transform_revenu_or_dette(X[['Dette']])
    X_clean_excel['Age_Clean'] = transform_age(X[['Age_Clean']])
    X_clean_excel['Employe_Statut'] = normaliser_employe(X[['Employe_Statut']])
    X_clean_excel['Ville_Clean'] = normaliser_ville(X[['Ville_Clean']])
    X_clean_excel['Device_Clean'] = normaliser_other(X[['Device_Clean']])

    # Ajoutons la colonne cible a la fin
    X_clean_excel['Fraude_Cible'] = y


    # Supprime les autres colonnes brutes
    colonnes_finales = [col for col in X_clean_excel.columns if not col.endswith('_raw')]
    df_final_export = X_clean_excel[colonnes_finales]

    # Exportation vers Excel
    df_final_export.to_excel("dataset_fraude_nettoye_final.xlsx", index=False)

    print("Fichier excel cree avec succes !")

# ==========================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ==========================================

# Chargements des donnees
df = pd.read_excel('dataset_pretraitement_fraude.xlsx')

# ==================================
# Detecter et Suppression les doublons
# ==================================

# Compter combien au total
nombre_doublons = df.duplicated(subset=['TransactionID']).sum()

print(f"Nombre de lignes dupliquées détectées : {nombre_doublons}")
df = df.drop_duplicates(subset=['TransactionID'])

print("On vient de supprimer les doublons !")
print("==============================================================\n\n")
# ===================================================================


col_exclude = ['TransactionID', 'ClientID', 'Commentaire']

col_todo_exclude = [col for col in df.columns if col.startswith('TODO_')]

X = df.drop(columns=col_exclude + col_todo_exclude)

y = df['Fraude']

# renommer les colonnes pour eviter les confusions entre les brutes et les pretraitees
X = rename_columns(X)

# ==================================================
# EXPORTATION DES DONNEES TRAITEES POUR EXPLORATION
# ==================================================
export_data_clean(X, y)


# ==========================================
# PIPELINE DE PRETRAITEMENT + MODELE
# ==========================================
niveau_etude_order = [
    'Primaire',
    'Secondaire',
    'Doctorat',
    'Master',
    'Licence',
    'Inconnu',
]

# On definit les colonnes pour eviter de se perdre
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        # 1. Montant : Transform, Impute, Scale
        ('num_montant', Pipeline([
            ('convert', FunctionTransformer(transform_montant)),
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler())
        ]), ['Montant_HTG']),

        # Dans ton preprocessing_pipeline :
        ('temporel', Pipeline([
            ('extract', FunctionTransformer(transform_date_features)), 
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['DateTransaction']),

        # 2. Anciennete : Transform, Impute, Scale
        ('num_anciennete', Pipeline([
            ('convert', FunctionTransformer(transform_anciennete_en_jours)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Anciennete_Jours']),

        # 3. Revenu & Dette : Transform, Impute, Scale j'utilise la meme fonction pour les deux
        ('num_revenu', Pipeline([
            ('convert', FunctionTransformer(transform_revenu_or_dette)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), ['Revenu_Mensuel']),

        ('num_dette', Pipeline([
            ('convert', FunctionTransformer(transform_revenu_or_dette)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Dette']),

        # 4. Age : Transform, Impute, Scale
        ('num_age', Pipeline([
            ('convert', FunctionTransformer(transform_age)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Age_Clean']),

        # 5. Employe j'ai deja converti en 0/1/NaN : Impute
        ('num_employe', Pipeline([
            ('convert', FunctionTransformer(normaliser_employe)),
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]), ['Employe_Statut']),

        # 6. Categories : Normaliser, OneHotEncoder
        ('cat_ville', Pipeline([
            ('norm', FunctionTransformer(normaliser_ville)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Ville_Clean']),

        ('cat_autres', Pipeline([
            ('norm', FunctionTransformer(normaliser_other)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Device_Clean']),

        # TypeMarchand est une variable cat -> OneHotEncode apres normalisation
        ('cat_type_marchand', Pipeline([
            ('norm', FunctionTransformer(normaliser_other)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['TypeMarchand']),

        # Canal est cat -> OneHotEncode apres normalisation
        ('cat_canal', Pipeline([
            ('norm', FunctionTransformer(normaliser_other)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Canal']),

        ('cat_niveau_etude', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord', OrdinalEncoder(categories=[niveau_etude_order], handle_unknown='use_encoded_value', unknown_value=-1))
        ]), ['NiveauEtude']),

        # 7. NbTrans_24h : Impute, Scale
        ('num_nbtrans_24h', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['NbTrans_24h']),

        # 8. StatutMarital : Impute, OneHotEncoder
        ('cat_statutmarital', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['StatutMarital']),

        # 9. Devise_indiquee : Normaliser, Impute, OneHotEncoder
        ('cat_devise', Pipeline([
            ('norm', FunctionTransformer(transform_device)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Devise']),
    ]
)

full_model = Pipeline([
    ('pretraitement', preprocessing_pipeline),
    ('modele', LogisticRegression(max_iter=1000, random_state=42))
])

# ==========================================
# 4. ENTRAINEMENT ET EVALUATION
# ==========================================

# Decoupage en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_model, X_train, y_train, cv=skf, scoring="accuracy")

print("--- Validation croisée (train uniquement) ---")
print(f"Précision moyenne : {cv_scores.mean():.2%}")
print(f"Écart-type : {cv_scores.std():.4f}")

# Entrainement du modele
full_model.fit(X_train, y_train)

# Prediction sur le jeu de test
y_pred = full_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Petit bonus : Matrice de confusion simplifiée
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nFraudes détectées (Vrais Positifs) : {tp}")
print(f"Fraudes ratées (Faux Négatifs)     : {fn}")
print(f"Clients innocents bloqués (Faux Positifs) : {fp}")
