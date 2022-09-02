import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc

df = pd.read_excel('/Users/andreannebernatchez/Documents/PSY_3008/Fichier_Concat3.xlsx')
df = df.drop(['cortex_vol_VolL_DK_x', 'cortex_vol_VolR_DK_x'], axis=1) #No values in these columns

''' Separates group data in different pandas.Dataframe
        Group 1: Parkinson
        Group 2: Control
        Group 3: Parkinson without dopaminergic deficit
'''

df_Group1 = df[df.APPRDX == 1]
df_Group2 = df[df.APPRDX == 2]
df_Group3 = df[df.APPRDX == 3]


def dictionary (lhrh_feat_d, regionR, regionL):

    ''' Creates a dictionary that includes a list of the same features that are in different hemispheres.
        Returns: 
            {'feature_name':('right-corresponding-feature', 'left-corresponding-feature')}
    '''

    var_r = [i for i in df.columns if regionR in i]
    for var in var_r: 
        common_feature_name = var.replace(regionR, '')
        feature = var
        contralateral_feature = var.replace(regionR, regionL)
        lhrh_feat_d [common_feature_name] = feature, contralateral_feature


class LateralityAnalysis():

    ''' Creates a pandas.DataFrame with laterality analysis using the formula:
            laterality = (feature - contralateral_feature) / (feature + contralateral_feature)
        Args: 
            df: pandas.DataFrame to be used for columns
            lhrh_feat_d: {'common_feature_name':('feature', 'contralateral_feature')}
            file_name: name of the file.csv
            PATH_save_results: abspath to save the file.csv
        Returns: 
            pandas.DataFrame csv file with results
    '''

    def __init__(self,
                    df,
                    lhrh_feat_d,
                    file_name,
                    PATH_save_results):

        self.df          = df
        self.lhrh_feat_d = lhrh_feat_d
        self.file_name   = file_name
        self.PATH_save   = PATH_save_results
        self.run()

    def run(self):

        df_lat = pd.DataFrame()
        for common_feature_name in self.lhrh_feat_d:
            feature = self.lhrh_feat_d[common_feature_name][0]
            contralateral_feature = self.lhrh_feat_d[common_feature_name][1]
            df_lat[common_feature_name] = (self.df[feature]-self.df[contralateral_feature]) / (self.df[feature] + self.df[contralateral_feature])
        df_lat.to_csv(os.path.join(self.PATH_save, f'{self.file_name}.csv'))


def plot_laterality_per_group(X,
                                group1_values,
                                group2_values,
                                group3_values,
                                parameter):

    ''' Creates a plot for laterality analysis
        Args:
            X: list() of pandas.DataFrame.columns
            group1_values: numpy.array of values for group 1
            group2_values: numpy.array of values for group 2
            group3_values: numpy.array of values for group 3
        Returns:
            Saves plot
    '''

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, group1_values, 0.2, label = 'Group 1')
    plt.bar(X_axis, group2_values, 0.2, label = 'Group 2')
    plt.bar(X_axis + 0.2, group3_values, 0.2, label = 'Group 3')
      
    plt.xticks(X_axis, X, rotation='vertical')
    plt.xlabel('Regions')
    plt.ylabel('Laterality Index')
    plt.title('Laterality Index by region, by group' + parameter)
    plt.grid(True, color = "grey", linewidth = "0.3", linestyle = "-")
    plt.legend()
    plt.show()


# 1. THICKNESS
#------------------
lhrh_feat_d_Thick = {}
dictionary(lhrh_feat_d_Thick, 
                            '_ThickR_DK_x', 
                            '_ThickL_DK_x')

# GROUP 1:
file_name_Thick_Group1 = 'Laterality_Analysis_Thickness_Group1'
PATH_save_results = '/Users/andreannebernatchez/Documents/PSY_3008'

LateralityAnalysis(df_Group1, 
                            lhrh_feat_d_Thick, 
                            file_name_Thick_Group1, 
                            PATH_save_results)


# GROUP 2:
file_name_Thick_Group2 = 'Laterality_Analysis_Thickness_Group2'

LateralityAnalysis(df_Group2, 
                            lhrh_feat_d_Thick, 
                            file_name_Thick_Group2, 
                            PATH_save_results)

# GROUP 3:
file_name_Thick_Group3 = 'Laterality_Analysis_Thickness_Group3'

LateralityAnalysis(df_Group3, 
                            lhrh_feat_d_Thick, 
                            file_name_Thick_Group3, 
                            PATH_save_results)

# 1.1 Average LI by region, by group
# Group 1:
Means_Thick_g1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group1.csv')
Means_Thick_g1.loc['Mean_Group_1'] = Means_Thick_g1.mean()

# Group 2:
Means_Thick_g2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group2.csv')
Means_Thick_g2.loc['Mean_Group_2'] = Means_Thick_g2.mean()

# Group 3:
Means_Thick_g3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group3.csv')
Means_Thick_g3.loc['Mean_Group_3'] = Means_Thick_g3.mean()

# 1.2 Bar plot
Means_table_Thick = pd.concat([Means_Thick_g1.tail(1), Means_Thick_g2.tail(1), Means_Thick_g3.tail(1)], axis = 0)
Means_table_Thick = Means_table_Thick.iloc[: , 1:]

X = Means_table_Thick.columns.tolist()
Group1_Thick = Means_table_Thick.values[0]
Group2_Thick = Means_table_Thick.values[1]
Group3_Thick = Means_table_Thick.values[2]

plot_laterality_per_group(X,
                        Group1_Thick,
                        Group2_Thick,
                        Group3_Thick,
                        ' (Thickness)')


# 2. VOLUME
#------------------
lhrh_feat_d_Vol = {}
dictionary(lhrh_feat_d_Vol, 
                            '_VolR_DK_x', 
                            '_VolL_DK_x')

# GROUP 1:
file_name_Vol_Group1 = 'Laterality_Analysis_Volume_Group1'

LateralityAnalysis(df_Group1, 
                            lhrh_feat_d_Vol, 
                            file_name_Vol_Group1, 
                            PATH_save_results)

# GROUP 2:
file_name_Vol_Group2 = 'Laterality_Analysis_Volume_Group2'

LateralityAnalysis(df_Group2, 
                            lhrh_feat_d_Vol, 
                            file_name_Vol_Group2, 
                            PATH_save_results)

# GROUP 3:
file_name_Vol_Group3 = 'Laterality_Analysis_Volume_Group3'

LateralityAnalysis(df_Group3, 
                            lhrh_feat_d_Vol, 
                            file_name_Vol_Group3, 
                            PATH_save_results)

# 2.1 Average LI by region, by group
# Group 1:
Means_Vol_g1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group1.csv')
Means_Vol_g1.loc['Mean_Group_1'] = Means_Vol_g1.mean()

# Group 2:
Means_Vol_g2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group2.csv')
Means_Vol_g2.loc['Mean_Group_2'] = Means_Vol_g2.mean()

# Group 3:
Means_Vol_g3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group3.csv')
Means_Vol_g3.loc['Mean_Group_3'] = Means_Vol_g3.mean()

# 2.2 Bar plot
Means_table_Vol = pd.concat([Means_Vol_g1.tail(1), Means_Vol_g2.tail(1), Means_Vol_g3.tail(1)], axis = 0)
Means_table_Vol = Means_table_Vol.iloc[: , 1:]

X = Means_table_Vol.columns.tolist()
Group1_Vol = Means_table_Vol.values[0]
Group2_Vol = Means_table_Vol.values[1]
Group3_Vol = Means_table_Vol.values[2]

plot_laterality_per_group(X,
                            Group1_Vol,
                            Group2_Vol,
                            Group3_Vol,
                            ' (Volume)')

# 3. AREA
# -----------------
lhrh_feat_d_Area = {}
dictionary(lhrh_feat_d_Area, 
                            '_AreaR_DK_x', 
                            '_AreaL_DK_x')

# GROUP 1:
file_name_Area_Group1 = 'Laterality_Analysis_Area_Group1'

LateralityAnalysis(df_Group1, 
                            lhrh_feat_d_Area, 
                            file_name_Area_Group1, 
                            PATH_save_results)

# GROUP 2:
file_name_Area_Group2 = 'Laterality_Analysis_Area_Group2'

LateralityAnalysis(df_Group2, 
                            lhrh_feat_d_Area, 
                            file_name_Area_Group2, 
                            PATH_save_results)

# GROUP 3:
file_name_Area_Group3 = 'Laterality_Analysis_Area_Group2'

LateralityAnalysis(df_Group3, 
                            lhrh_feat_d_Area, 
                            file_name_Area_Group3, 
                            PATH_save_results)

# 3.1 Average LI by region, by group
# Group 1:
Means_Area_g1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group1.csv')
Means_Area_g1.loc['Mean_Group_1'] = Means_Area_g1.mean()

# Group 2:
Means_Area_g2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group2.csv')
Means_Area_g2.loc['Mean_Group_2'] = Means_Area_g2.mean()

# Group 3:
Means_Area_g3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group3.csv')
Means_Area_g3.loc['Mean_Group_3'] = Means_Area_g3.mean()

# 3.2 Bar plot
Means_table_Area = pd.concat([Means_Area_g1.tail(1), Means_Area_g2.tail(1), Means_Area_g3.tail(1)], axis = 0)
Means_table_Area = Means_table_Area.iloc[: , 1:]

X = Means_table_Area.columns.tolist()
Group1_Area = Means_table_Area.values[0]
Group2_Area = Means_table_Area.values[1]
Group3_Area = Means_table_Area.values[2]

plot_laterality_per_group(X,
                            Group1_Area,
                            Group2_Area,
                            Group3_Area,
                            ' (Area)')

'''
    Creates pandas.dataframe that is going to be used in def ANOVA
    ** Note : à raccourcir
'''
    
df_T1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group1.csv')
df_T1.insert(0, 'Groups', 1)
df_T2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group2.csv')
df_T2.insert(0, 'Groups', 2)
df_T3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Thickness_Group3.csv')
df_T3.insert(0, 'Groups', 3)

df_T = pd.concat([df_T1, df_T2, df_T3], ignore_index = True)
df_T = df_T.drop(['Unnamed: 0'], axis = 1)

df_V1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group1.csv')
df_V1.insert(0, 'Groups', 1)
df_V2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group2.csv')
df_V2.insert(0, 'Groups', 2)
df_V3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Volume_Group3.csv')
df_V3.insert(0, 'Groups', 3)

df_V = pd.concat([df_V1, df_V2, df_V3], ignore_index = True)
df_V = df_V.drop(['Unnamed: 0'], axis = 1)

df_A1 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group1.csv')
df_A1.insert(0, 'Groups', 1)
df_A2 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group2.csv')
df_A2.insert(0, 'Groups', 2)
df_A3 = pd.read_csv('/Users/andreannebernatchez/Documents/PSY_3008/Laterality_Analysis_Area_Group3.csv')
df_A3.insert(0, 'Groups', 3)

df_A = pd.concat([df_A1, df_A2, df_A3], ignore_index = True)
df_A = df_A.drop(['Unnamed: 0'], axis = 1)


def ANOVA (df,
            df_means):

    ''' ANOVA (checking for significant differences between groups)
        Args: 
            df: pandas.Dataframe to be used
        Returns:
            ANOVA results
    '''

    keys = []
    tables = []

    for variable in df.columns[1:]:
        model = ols('{} ~ Groups'.format(variable), data = df).fit()

        # ASSUMPTION CHECK
        # 1. Independence (all groups must be mutually exclusives)

        # 2. Normality 
        print(stats.shapiro(model.resid))

        # 3. Homogeneity of variance
        print(stats.levene(df[variable][df['Groups'] == 1], df[variable][df['Groups'] == 2], df[variable][df['Groups'] == 3]))

        anova_table = sm.stats.anova_lm(model, typ = 2)

        keys.append(variable)
        tables.append(anova_table)

        if anova_table['PR(>F)'].values[0] < 0.05:
            print(variable)
            comp = mc.MultiComparison(df[variable], df['Groups'])
            post_hoc = comp.tukeyhsd()
            print(post_hoc.summary())

    df_anova = pd.concat(tables, keys = keys, axis = 0)
    print(df_anova)

    results_table = pd.DataFrame(
        data = df_means.values,
        columns = df_means.columns,
        index = ['Mean_LI_group1', 
                                'Mean_LI_group2', 
                                'Mean_LI_group3']) 
    #results_table.columns = pd.MultiIndex.from_product([['Regions'], results_table.columns]) - Multiindex not working when saving in .csv file
    results_table.loc['p_value'] = df_anova['PR(>F)'][::2].values
    for index, value in enumerate(results_table.iloc[3]):
        if value < 0.05:
            value = str(value)
            sig = ' *'
            results_table.iat[3, index] = value + sig
    results_table.to_csv(os.path.join(PATH_save_results, f'{file_name}.csv'))

file_name = 'Laterality_Analysis_Results_Thickness'
ANOVA(df_T, Means_table_Thick)

file_name = 'Laterality_Analysis_Results_Volume'
ANOVA(df_V, Means_table_Vol)

file_name = 'Laterality_Analysis_Results_Area'
ANOVA(df_A, Means_table_Area)
