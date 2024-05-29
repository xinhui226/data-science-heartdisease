import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import matplotlib.patches as mpatches
from PIL import Image

heart_disease = fetch_ucirepo(name='Heart Disease')

da = pd.read_csv(heart_disease.metadata.data_url)
df = da.dropna()
df.rename(
    columns={
        'num':'target',
        'cp':'chest pain',
        'trestbps':'resting blood pressure (mm/hg)',
        'chol':'cholesterol mg/dl',
        'fbs':'fasting blood sugar (> 120 mg/dl)',
        'restecg':'rest electrocardiographic result',
        'thalach':'max-achieved heart rate',
        'exang':'exercise-induced angina',
        'oldpeak':'exercise-induced ST',
        'slope':'slope of peak exercise ST',
        'ca':'num of major vessels',
        'thal':'thalassemia',
    },
    inplace=True)

df['num of major vessels'] = df['num of major vessels'].astype('int')
df['target'] = df['target'].map(lambda x: "No Disease" if x == 0 else 'Disease')
df['sex'] = df['sex'].replace({1: "Male", 0: "Female"})
df['chest pain'] = df['chest pain'].replace({1: "typical_angina", 
                          2: "atypical_angina", 
                          3:"non-anginal pain",
                          4: "asymtomatic"})
df['fasting blood sugar (> 120 mg/dl)'] = df['fasting blood sugar (> 120 mg/dl)'].replace({1: "True", 0: "False"})
df['exercise-induced angina'] = df['exercise-induced angina'].replace({1: "Yes", 0: "No"})
df['slope of peak exercise ST'] = df['slope of peak exercise ST'].replace({1: "upsloping", 2: "flat",3:"downsloping"})
df['thalassemia'] = df['thalassemia'].replace({3:"normal", 6: "fixed_defect", 7: "reversable_defect"})
df['rest electrocardiographic result X'] = df['rest electrocardiographic result'].replace({0:"normal", 1: "abnormal", 2: "hyper"})

continous_cols = ['age','resting blood pressure (mm/hg)','cholesterol mg/dl','max-achieved heart rate','exercise-induced ST']

for col in continous_cols:
    col_data = df[col]
    Q1 = np.percentile(col_data, 25.)
    Q3 = np.percentile(col_data, 75.)
    IQR = Q3-Q1
    outlier_step = IQR * 1.5
    outliers = col_data[~((col_data >= Q1 - outlier_step) & (col_data <= Q3 + outlier_step))].index.tolist()  
    df.drop(outliers,inplace=True)

X = df.drop(['target','rest electrocardiographic result X'],axis=1)
y = df['target']

encoder = LabelEncoder()

encode_cols = ['sex','chest pain','fasting blood sugar (> 120 mg/dl)','exercise-induced angina','slope of peak exercise ST','thalassemia']
for col in encode_cols:
    X[col] = encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

def calc_plot_text(axe, series, font = 12, h = 10):
        total = series.value_counts().sum()

        for patch in [patch for patch in axe.patches if patch.get_height() > 0]:
            x = patch.get_x() + (patch.get_width() / 2)
            height = patch.get_height()
            label = f"{round((height / total) * 100, 2)}%"
            y = height - h
            axe.text(x, y, label, ha='center', va='top', fontsize=font, color='white', weight='bold')

with open('heartdisease_lr_model.pkl','rb') as f:
    model = pickle.load(f)
    
st.set_page_config(page_title='Heart Disease App', layout='centered')

st.sidebar.title('Heart Disease Project')
pages = ['Introduction', 'Visualization', 'Prediction']
page = st.sidebar.selectbox('Choose a page',pages)



#  Introduction #
if page == 'Introduction':
    st.title('Heart Disease Project')

    st.write('***This project aims to analyze the UCI Heart Disease dataset and develop a machine learning model for predicting the presence or absence of heart disease.***')

    st.subheader('Importance of Heart Disease Prediction')
    st.write('Based on the [World Health Organization (WHO) report](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)) in 2020, heart disease is a leading cause of death worldwide. Early detection and prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project highlights the significance of leveraging data analysis and machine learning techniques for predicting heart disease risk.')
    st.write('According to the WHO:')
    st.write('- [Cardiovascular diseases are the leading cause of death globally](https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1), accounting for an estimated 17.9 million deaths per year.')
    st.write('- [About 85% of all cardiovascular disease deaths are due to heart attacks and strokes](https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/cardiovascular-diseases).')
    st.write('- [Early detection and management of risk factors can significantly reduce the burden of heart disease](https://www.who.int/activities/preventing-cardiovascular-disease).')

    st.subheader('Dataset Description')
    st.write('The UCI Heart Disease dataset contains 303 instances and 76 attributes. However, for this analysis, we focused on a subset of 14 relevant attributes.')

    st.write('**Features Included**')
    st.write('***age*** - Age')
    st.write('***sex*** - Sex')
    st.write('***cp*** - Chest pain type')
    st.write('***trestbp***- Resting blood pressure')
    st.write('***chol*** - Serum cholesterol levels')
    st.write('***fbs*** - Fasting blood sugar')
    st.write('***restecg*** - Resting electrocardiographic results')
    st.write('***thalach*** - Maximum heart rate achieved')
    st.write('***exang*** - Exercise-induced angina')
    st.write('***oldpeak*** - ST depression induced by exercise')
    st.write('***slope*** - Slope of the peak exercise ST segment')
    st.write('***ca*** - Number of major vessels colored by fluoroscopy')
    st.write('***thal*** - Thalassemia')
    st.write('***num*** - Target')
    st.write('***For more information about features please refer to below***')
    
    st.subheader('Dataset Preview')
    st.write(da.sample(10))
    
    st.subheader('Methodology')
    st.write('The project followed a systematic approach, including:')
    st.write('- Data preprocessing techniques (e.g., handling missing values, remove outliers)')
    st.write('- Exploratory data analysis (EDA) to gain insights into the dataset')
    st.write('- Feature selection techniques to identify the most relevant features')
    st.write('- Application of machine learning algorithms such as logistic regression, decision trees, and random forests to train machine learning model for heart disease prediction')
    st.write('- Utilization of scikit-learn pipelines to streamline the data processing and model training workflow')
    st.write('- Hyperparameter tuning using grid search and cross-validation to optimize model performance')
    st.write('- Implementation of a multilayer perceptron (MLP) neural network for heart disease classification')

    st.subheader('Usage Instructions')
    st.write('- Navigate to the ***Visualization*** page to view the results of the analysis performed on numerical variables, categorical variables, and the correlation analysis between features.')
    st.write('- Additionally, you can explore the ***Prediction*** page to input relevant feature values and obtain a prediction for the presence or absence of heart disease based on the trained machine learning model.')

    st.subheader('Acknowledgments')
    st.write('This project utilized the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).')

    st.subheader('Contact Information')
    st.write('For more information or feedback, please contact me at **xinhui22c@forward.edu.my** .')
    
    st.write('---')
    
    st.subheader('Features Explaination')
    st.image(Image.open('cp.png'),caption='Chest Pain Explaination')
    st.image(Image.open('restecg.png'),caption='Resting Electrocardiographic Results Explaination')
    st.image(Image.open('exerang.png'),caption='Exercise-induced Angina Explaination')
    st.image(Image.open('exerST.png'),caption='ST Depression Induced by Exercise Explaination')
    st.image(Image.open('slopeexerST.png'),caption='Slope of the peak Exercise ST Segment Explaination')
    st.image(Image.open('nummajorvessel.png'),caption='Number of Major Vessels Colored by Fluoroscopy Explaination')
    st.image(Image.open('thal.png'),caption='Thalassemia Explaination')
#  Introduction #

    
    
#  Visualization #
elif page=='Visualization':
    st.title('Visualization')
    st.write('This page focuses on visualizing the Heart Disease dataset and presenting the results of the exploratory data analysis. The visualizations provide insights into the distribution of numerical variables, the breakdown of categorical variables, and the correlation between different features.')
    
    # Numerical Variable #
    st.write('### Numerical Variables')
    num_feats = ['age', 'cholesterol mg/dl', 'resting blood pressure (mm/hg)', 'max-achieved heart rate', 'exercise-induced ST', 'num of major vessels']
    ncol= 2
    nrow= int(np.ceil(len(num_feats)/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 12))
    for i, col in enumerate(num_feats):
        plt.subplot(nrow, ncol, i + 1)
        
        ax = sns.kdeplot(data=df, x=col, hue="target", fill=True, alpha=.5, palette='Set2') 
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("density", fontsize=12)
        sns.despine(right=True)
        sns.despine(offset=0, trim=False)
        
        if col == 'num of major vessels':
            sns.countplot(data=df, x=col, hue="target", palette='Set2')
            calc_plot_text(ax, df[col], font=8, h=1)
    plt.gcf().text(.4, 1.02, 'Distribution of Numerical Features', style='oblique', size=14)
    plt.tight_layout()
    st.pyplot(fig)
    st.write('**Age Distribution**: There is a slight shift towards higher ages for patients with heart disease.')
    st.write('**Cholesterol Levels**: Patients with heart disease tend to have higher cholesterol levels compared to those without heart disease.')
    st.write('**Resting Blood Pressure**: Patients with heart disease generally have higher resting blood pressure values.')
    st.write('**Maximum Achieved Heart Rate**: Patients with heart disease tend to have lower maximum achieved heart rates compared to those without heart disease.')
    st.write('**Exercise-induced ST**: Shows a peak around 0 for both classes, but with a slightly higher frequency of non-zero values for patients with heart disease.')
    st.write('**Number of Major Vessels**: Patients with heart disease tend to have a higher number of affected major vessels.')

    # Categorical Variable #
    st.write('### Categorical Variables')
    fig, axes = plt.subplots(3,2,figsize=(12,12))
    plt.gcf().text(.4, 1.02, 'Distribution of Categorical Features', style='oblique', size=14)

    axes0 = sns.countplot(df, x='sex',hue='target',gap=.1,palette='Set2',ax=axes[0,0])
    axes0.set_xlabel('Gender',fontsize=12)
    calc_plot_text(axes0, df['sex'],font=11)

    axes1 = sns.countplot(df, x='fasting blood sugar (> 120 mg/dl)',hue='target',gap=.1,palette='Set2',ax=axes[0,1])
    axes1.set_xlabel('Higher Fasting Blood Sugar',fontsize=12)
    axes1.set_xticks(ticks=[0,1], labels=['Greater than (>) 121 mg/dl)', 'Lesser than (<) 121 mg/dl)'])
    calc_plot_text(axes1, df['fasting blood sugar (> 120 mg/dl)'],h=6,font=11)

    axes2 = sns.countplot(df, x='exercise-induced angina',hue='target',gap=.1,palette='Set2',ax=axes[1,0])
    calc_plot_text(axes2, df['exercise-induced angina'],font=11)
    axes2.set_xlabel('Exercise-Induced Angina',fontsize=12)

    axes3 = sns.countplot(df, x='chest pain',hue='target',gap=.1,palette='Set2',ax=axes[1,1])
    calc_plot_text(axes3, df['chest pain'], font=9, h=3)
    axes3.set_xlabel('Chest Pain Type',fontsize=12)

    axes4 = sns.countplot(df, x='slope of peak exercise ST',hue='target',gap=.1,palette='Set2',ax=axes[2,0])
    calc_plot_text(axes4, df['slope of peak exercise ST'], font=10, h=3)
    axes4.set_xlabel('Slope of the peak Exercise ST segment',fontsize=12)

    axes5 = sns.countplot(df, x='thalassemia',hue='target',gap=.1,palette='Set2',ax=axes[2,1])
    calc_plot_text(axes5, df['thalassemia'], font=9, h=2)
    axes5.set_xlabel('Thalassemia',fontsize=12)

    sns.despine(right=True)
    sns.despine(offset=0, trim=False)
    
    st.pyplot(fig)
    st.write('**Gender**: A higher proportion of males compared to females have heart disease in this dataset.')
    st.write('**Higher Fasting Blood Sugar**: Patients with lower fasting blood sugar levels have a higher prevalence of heart disease compared to those with higher fasting blood sugar levels.')
    st.write('**Exercise-Induced Angina**: A larger percentage of patients who experience exercise-induced angina (chest pain during exercise) have heart disease compared to those who do not experience angina during exercise.')
    st.write('**Chest Pain Type**: Non-anginal pain is the most common chest pain type for patients with heart disease, with 33.09% of patients in this class experiencing this type of chest pain.')
    st.write('**Slope of the Peak Exercise ST Segment**: Patients with a downsloping or flat slope of the peak exercise ST segment appear to be more likely to have heart disease compared to those with an upsloping slope.')
    st.write('**Thalassemia**: A higher proportion of patients with reversible defects (reversible thalassemia) seem to have heart disease compared to those with fixed defects or normal thalassemia.')
    
    # Correlation Variable #
    st.write('### Correlation Analysis')
    
    # Num of Vessel vs Chest Pain #
    st.write('**Correlation Analysis between Number of Major Vessels and Chest Pain Type**')
    cat_fig = sns.catplot(df, x="num of major vessels", hue="target", palette='Set2', col="chest pain", kind='count')
    plt.gcf().text(.4, 1.05, 'Number of major vessels vs Chest pain type', style='oblique', size=14)
    st.pyplot(cat_fig)
    st.write("For the patients experiencing typical angina chest pain, the count is highest when there are 0 major vessels affected, suggesting no significant blockage in major vessels for many of these cases.")
    
    st.write('')
    
    # Exercise-induced Angina vs Resting ECG #
    st.write('**Exercise-induced Angina Distribution by Resting Electrocardiographic Results**')
    df_eda3 = df[['exercise-induced angina', 'rest electrocardiographic result X']]
    df_eda3 = pd.DataFrame(df_eda3.groupby(['exercise-induced angina', 'rest electrocardiographic result X']).size().reset_index(name='total'))
    df_eda3_0 = df_eda3[df_eda3['rest electrocardiographic result X'] == 'normal'].drop('rest electrocardiographic result X', axis=1)
    df_eda3_1 = df_eda3[df_eda3['rest electrocardiographic result X'] == 'abnormal'].drop('rest electrocardiographic result X', axis=1)
    df_eda3_2 = df_eda3[df_eda3['rest electrocardiographic result X'] == 'hyper'].drop('rest electrocardiographic result X', axis=1)

    total_list = [df_eda3_0['total'], df_eda3_1['total'], df_eda3_2['total']]
    ecg_types = ['normal','abnormal','hyper']
    suptitle = dict(x=0.5, y=0.94, fontsize=14, weight='heavy', ha='center', va='center')
    exp_text = dict(x=0.5, y=0.17, fontsize=6, weight='normal', ha='center', va='center', textalign='center')
    highlight_explanation = [{'weight':'bold', 'color': 'gainsboro'}, {'weight':'bold', 'color': 'gainsboro'}, {'weight':'bold', 'color': 'cadetblue'}]
    l_120mg = mpatches.Patch(color='gainsboro', label='Exercise-induced angina (False)')
    m_120mg = mpatches.Patch(color='cadetblue', label='Exercise-induced angina (True)')

    def display_eda3(subplot_num, ecg_type, total, colors, start_angle):
        centre = plt.Circle((0, 0), 0.85, fc='white', edgecolor='black', linewidth=0.5)
        total_patients = total.sum()
        
        plt.subplot(1, 3, subplot_num)
        plt.tight_layout(rect=[0, 0, 1, 1.01])
        plt.pie(total, colors=colors, autopct='%.2f%%', pctdistance=0.65, startangle=start_angle, wedgeprops=dict(alpha=0.85, edgecolor='black', linewidth=0.5), textprops={'fontsize': 7})
        plt.text(0, 0.08, ecg_types[ecg_type], weight='bold', ha='center', fontsize=10)
        plt.text(0, -0.08, f"{total_patients} patients", ha='center', fontsize=8)
        fig=plt.gcf()
        fig.gca().add_artist(centre)

    donut_fig = plt.figure(figsize=(9, 4))
    for idx, total in enumerate(total_list):
        display_eda3(idx+1, idx, total, ['gainsboro', 'cadetblue'], 20)
        if idx == 1: plt.legend(handles=[l_120mg, m_120mg], loc='upper center', bbox_to_anchor=(0.5, .2), ncol=2, borderpad=3, frameon=False, fontsize=7, columnspacing=3)
    plt.gcf().text(.3, .8, 'Exercise-induced Angina Distribution by Resting Electrocardiographic Results', style='italic', fontsize=7)
    st.pyplot(donut_fig)
    st.write('It suggests that patients with normal and hyper resting electrocardiographic readings are more likely not experience angina during exercise.')
    
    st.write('')
    
    # Resting Blood Pressure vs. Age by Heart Disease #
    st.write('**Resting Blood Pressure vs Age by Heart Disease**')
    
    rbp_mean, age_mean = round(df['resting blood pressure (mm/hg)'].mean(),2), round(df['age'].mean(),2)
    scatter_fig, scatter_axes = plt.subplots(figsize=(10,5))
    sns.scatterplot(df, x='age', y='resting blood pressure (mm/hg)', ax=scatter_axes, hue='target', palette='Set2')
    plt.grid(True, ls='--', lw=.3)
    plt.axhline(rbp_mean, lw=.3, color='k', ls='--')
    plt.text(27, rbp_mean * 0.99, va='top', ha='left', s=f'Mean of Resting blood pressure: \n {rbp_mean}', size='x-small')
    plt.axvline(age_mean, lw=.3, color='k', ls='--')
    plt.text(age_mean * 1.01, 96, va='top', ha='left', s=f'Mean of Age: \n {age_mean}', size='x-small')
    plt.legend(title='target', bbox_to_anchor=(1.35, .6), borderaxespad=.5)
    plt.gcf().text(.2, .93, 'Resting Blood Pressure vs. Age by Heart Disease', style='oblique', size=12)
    
    st.pyplot(scatter_fig)
    st.write('There is a general upward trend in resting blood pressure with increasing age, which is expected. However, there is considerable overlap between the two groups, suggesting that resting blood pressure alone may not be a strong predictor of heart disease.')
    
    st.write('')
    
    # Fasting Blood Sugar vs. Cholesterol by Heart Disease #
    st.write('**Fasting Blood Sugar vs Cholesterol by Heart Disease**')
    
    violin_fig, violin_axes = plt.subplots(figsize=(10,5))
    sns.violinplot(
        x = "fasting blood sugar (> 120 mg/dl)",
        y = "cholesterol mg/dl",
        data=df,
        ax=violin_axes,
        hue='target',
        split=True,
        palette='Set2'
    )
    plt.gcf().text(.3, .95, 'Fasting Blood Sugar vs. Cholesterol', style='oblique', size=12)
    plt.xlabel('Higher Fasting Blood Sugar (> 120 mg/dL)')
    plt.ylabel('Cholesterol (mg/dL)')
    plt.legend(title='target', bbox_to_anchor=(1.35, .6), borderaxespad=.5)
    
    st.pyplot(violin_fig)
    st.write('Individuals with heart disease tend to have higher fasting blood sugar levels (more data points on the right side of the plot) and higher cholesterol levels (wider violin shape on the right side).')
    
    st.write('')
    
    # Exercise-induced ST depression Distribution by Slope of peak exercise ST segment #
    st.write('**Exercise-induced ST depression Distribution by Slope of peak exercise ST segment**')
    
    fig = plt.figure(figsize=(14,4))
    ax1 = plt.subplot(131)
    ax1.set_title('slope of peak exercise ST (Downsloping)')
    sns.histplot(data=df, x=df[df['slope of peak exercise ST'] == 'downsloping']["exercise-induced ST"], hue="target", multiple="dodge", shrink=.8, palette='Set2')

    ax2 = plt.subplot(132)
    ax2.set_title('slope of peak exercise ST (Flat)')
    sns.histplot(data=df, x=df[df['slope of peak exercise ST'] == 'flat']["exercise-induced ST"], hue="target", multiple="dodge", shrink=.8, palette='Set2')

    ax3 = plt.subplot(133)
    ax3.set_title('slope of peak exercise ST (Upsloping)')
    sns.histplot(data=df, x=df[df['slope of peak exercise ST'] == 'upsloping']["exercise-induced ST"], hue="target", multiple="dodge", shrink=.8, palette='Set2')

    plt.gcf().text(.3, 1.08, 'Exercise-induced ST depression Distribution by Slope of peak exercise ST segment', style='oblique', size=12)

    st.pyplot(fig)
    st.write('Individuals with heart disease (Disease group) tend to have a higher incidence of downsloping and flat ST segments, which are considered abnormal and indicative of potential cardiac issues.')
    
    st.write('---')

    st.subheader("Insights throughout the Analysis")
    st.write("""
            - Heart disease patients tend to have lower maximum heart rates achieved during exercise.
            - Individuals with heart disease are more likely to experience typical angina symptoms and angina induced by exercise.
            - A higher percentage of males are affected by heart disease compared to females.
            - Higher resting blood sugar levels and lower fasting blood sugar levels are associated with a higher incidence of heart disease.
            - Individuals with heart disease tend to have higher fasting blood sugar levels and higher cholesterol levels compared to those without heart disease.
            - There is a positive correlation between resting blood pressure and age, with higher blood pressure values observed in older individuals, particularly those with heart disease.
            - There is a positive correlation between the number of major vessels affected and the presence of typical angina symptoms. As the number of affected vessels increases, the likelihood of experiencing typical angina symptoms also rises.
            """)
#  Visualization #



#  Prediction #    
elif page=='Prediction':
    st.title('Prediction')
    st.write('This page allows you to input relevant feature values and obtain a prediction for the presence or absence of heart disease based on the trained machine learning model.')
    st.write('Please fill in the input fields below with the appropriate values, and the model will provide a prediction.')

    
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=0, max_value=130, value=0)
    restbp = st.number_input("Resting Blood Pressure (mm/hg)", min_value=0, max_value=300)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=300)
    maxheartrate = st.number_input("Max-achieved Heart Rate", min_value=30, max_value=280)
    exerciseST = st.number_input("Exercise-induced ST",step=0.1)
    chestpain = st.radio("Chest Pain", ["typical_angina", "atypical_angina", "non-anginal pain", "asymtomatic"])
    restecg = st.radio("Rest Electrocardiographic Result", ['Normal', 'Abnormal', 'Hyper'])
    SOPexerciseST = st.radio("Slope of Peak Exercise-induced ST", ['upsloping', 'flat', 'downsloping'])
    thal = st.radio("Thalassemia", ['normal', 'fixed_defect', 'reversable_defect'])
    nummajorvessel = st.slider("Num of Major Vessels", 0, 3, 0)
    fastingbp = st.checkbox("Fasting Blood Sugar over 120 mg/dl")
    exerciseangina = st.checkbox("Exercise-induced Angina")

    if st.button('Predict'):
        if fastingbp: fastingbp = "True"
        else: fastingbp = "False"
        if exerciseangina: exerciseangina = "Yes"
        else: exerciseangina = 'No'
        
        restecgresult = {'Normal':0, 'Abnormal':1, 'Hyper':2}
    
        input_data = [[age, gender, chestpain, restbp, chol, fastingbp, restecgresult[restecg], maxheartrate, exerciseangina, exerciseST, SOPexerciseST, nummajorvessel, thal]]
        for col in [1,2,5,8,10,12]:
            input_data[0][col] = encoder.fit_transform([input_data[0][col]])[0]
        input_scaled = scaler.transform(input_data)
        st.write(f'Predicted: {"Disease" if model.predict(input_scaled)[0] == 1 else "No Disease"}')
#  Prediction #