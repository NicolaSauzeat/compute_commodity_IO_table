import pandas as pd
import os
import numpy as np
import pdb

def importation_table(path, path_transition):
    #pdb.set_trace()
    use_table = pd.read_excel("./data/Use_IO.xlsx",sheet_name="2012", header=5, index_col="Code")
    transition_table = pd.read_excel("./data/matrice_transition_finale.xlsx", index_col="Code")
    make_table = pd.read_excel("./data/IO_US.xlsx", sheet_name="2012", header= 5, index_col="Code")
    output_sector = use_table.loc["T018"][0:406]
    tax_use = use_table.loc["T00TOP"][0:406]
    use_table.drop(columns="Commodity Description",index= ["T005","VABAS","VAPRO","T00TOP","T00SUB","V00300","T00OTOP","V00100","Note.  Detail may not add to total due to rounding.","T018"],inplace=True)
    return use_table, transition_table, make_table, output_sector, tax_use



def preprocess_use_table(use_table):
    input_output = use_table.iloc[0:405,0:405]
    input_output.fillna(0,inplace=True)
    input_output.loc['S00202']= 0.0
    for i in ['S00300', 'S00401', 'S00402', 'S00900']:
        input_output[i] = 0.0
    input_output.index = input_output.index.astype("str")
    input_output.columns = input_output.columns.astype("str")
    return input_output


def preprocess_transition_table(transition_table):
    transition_table.fillna(0,inplace=True)
    
    transition_table = transition_table.rename(index={"S00101":"S00202"})
    transition_table.dropna(inplace=True)
    transition_table.drop(index= ["T005","VABAS","VAPRO","T00TOP","T00SUB","V00300","T00OTOP","V00100","Note.  Detail may not add to total due to rounding.","T018"],inplace=True)
    transition_table = transition_table.groupby(transition_table.index.values,axis=0).sum()
    transition_table = transition_table.iloc[0:405]
    #transition_table.loc['S00900']= 0.0
    #transition_table.loc['S00900',"World Adjustment"]= 1
    transition_table.index = transition_table.index.astype("str")
    transition_table.columns = transition_table.columns.astype("str")
    transition_table.fillna(0.0, inplace=True)
    return transition_table

def preprocess_make_table(make):
    make.columns = make.columns.astype(str)
    make.fillna(0.0, inplace=True)
    make = make.drop(columns="Commodity Description")
    make = make.iloc[0:405,0:405]
    make.index = make.index.astype(str)
    make.columns = make.columns.astype(str)
    make.rename(columns={"331314":"331313"}, inplace=True)
    make = make.groupby(make.columns.values, axis=1).sum()
    for i in ['S00202']:
        make.loc[i]= 0.0
    return make

def preprocess_output_vector(output_sector, transition_table):
    output_sector.index = output_sector.index.astype("str")
    output_sector = output_sector.rename(index={"331314":"331313"})
    output_sector = output_sector.groupby(output_sector.index.values, axis=0).sum()
    output_sector.drop(index=["Commodity Description","814000","813B00","813A00","813100"], inplace= True)
    output_sector.index = output_sector.index.astype("str")
    transition_table.loc["S00201","84.13"]= 1.0
    transition_table.loc["S00101","84.13"]= 1.0
    transition_table= transition_table.drop(index=["S00401","S00402","S00300","814000","813B00","813A00","813100"])
    transition_table.fillna(0.0, inplace=True)
    output_sector_naf = output_sector.transpose().dot(transition_table)
    output_sector_naf.fillna(0.0,inplace=True)
    return output_sector_naf 


def transition_tax(tax_product,transition_table):
    tax_product.drop(index="Commodity Description", inplace=True)
    tax_product.fillna(0.0, inplace=True)
    tax_product.index= tax_product.index.astype(str)

    tax_product = tax_product.rename(index={"331314":"331313"})
    tax_product = tax_product.groupby(tax_product.index.values, axis=0).sum()    
    #transition = transition_table.drop(index=["S00203","S00401","S00402","S00300","S00900"])
    transition_tax = tax_product.transpose().dot(transition_table)
    transition_tax.fillna(0.0,inplace=True)
    return transition_tax
    


def transition_use_bea_naf(make, transition):
    transition= transition.drop(index=["S00401","S00402","S00300","814000","813B00","813A00","813100"])
    make = make.drop(index=["S00900","S00300","S00401","S00402","814000","813B00","813A00","813100"], columns=["814000","813B00","813A00","813100",'S00300', 'S00401', 'S00402', 'S00900'])
    nace_bea_matrix =make.transpose().dot(transition)
    nace_bea_matrix.index = nace_bea_matrix.index.astype("str")
    nace_bea_matrix.columns = nace_bea_matrix.columns.astype("str")
    nace_bea_matrix.rename(index={"331314":"331313"}, inplace=True)
    nace_bea_matrix = nace_bea_matrix.groupby(nace_bea_matrix.index.values, axis=0).sum()
    nace_bea_matrix.loc["S00201","84.13"]= 1.0
    nace_bea_matrix.loc["S00101","84.13"]= 1.0
    nace_bea_matrix.fillna(0.0,inplace=True)
    transition.loc["S00201","84.13"]= 1.0
    transition.loc["S00101","84.13"]= 1.0
    #nace_bea_matrix = nace_bea_matrix.drop(index=[['S00300', 'S00401', 'S00402', 'S00900']])
    transition.fillna(0.0,inplace=True)
    final_use_naf = nace_bea_matrix.transpose().dot(transition)    
    final_use_naf.fillna(0.0,inplace=True)
    return final_use_naf



def transition_make_bea_naf(make, transition_matrix):
    transition = transition_matrix.copy()
    transition.loc["S00201","84.13"]= 1.0
    transition.loc["S00101","84.13"]= 1.0
    transition= transition.drop(index=["S00401","S00402","S00300","814000","813B00","813A00","813100"])
    make = make.drop(index=["S00900","S00300","S00401","S00402","814000","813B00","813A00","813100"], columns=["814000","813B00","813A00","813100"])
    transition.fillna(0.0,inplace=True)
    nace_bea_matrix = make.dot(transition)
    nace_bea_matrix.index = nace_bea_matrix.index.astype("str")
    nace_bea_matrix.columns = nace_bea_matrix.columns.astype("str")
    nace_bea_matrix.rename(index={"331314":"331313"}, inplace=True)
    nace_bea_matrix = nace_bea_matrix.groupby(nace_bea_matrix.index.values, axis=0).sum()
    nace_bea_matrix.loc["S00201"]= 0.0
    nace_bea_matrix.loc["S00101"]= 0.0
    nace_bea_matrix.fillna(0.0,inplace=True)
    final_make_naf = nace_bea_matrix.transpose().dot(transition)    
    final_make_naf.fillna(0.0,inplace=True)

    return final_make_naf

def coeff_output(table):
    sum_output = table.sum()
    D = table / sum_output
    D.fillna(0.0, inplace=True)
    return D

def coeff_intermediate_consumption(use_table_naf,output_use_naf):
    intermediate_consumption = use_table_naf / output_use_naf
    intermediate_consumption.fillna(0.0, inplace=True)
    return intermediate_consumption

def calculate_tech_coeff(product, use):
    technical_coefficient= use.dot(product)
    technical_coefficient.fillna(0.0,inplace=True)
    for row in technical_coefficient.index:
        for column in technical_coefficient.columns:
            if technical_coefficient.loc[row,column]<0:
                technical_coefficient.loc[row,column]=0

    return technical_coefficient

def calculate_tech_coeff_comm(product_table, use_table):
    #pdb.set_trace()
    tech_coeff_comm=  use_table /product_table.transpose()
    tech_coeff_comm.replace([np.inf, -np.inf,np.nan], 0.0, inplace=True)
    return tech_coeff_comm



def leontief_commodity_tech_assumption(make_table_naf, use_table_naf):
    technical_coeff = calculate_tech_coeff_comm(make_table_naf,use_table_naf)
    identity_matrix= np.identity(616)
    leontief_commodity_technological_assump = pd.DataFrame(np.linalg.inv(identity_matrix - technical_coeff), index= technical_coeff.index, columns= technical_coeff.columns)
    indirect_consumption_comm_tech_assump = leontief_commodity_technological_assump - np.identity(616)
    return leontief_commodity_technological_assump, indirect_consumption_comm_tech_assump 




def leontief_industrial_tech_assumption(product, use):
    technical_coeff = calculate_tech_coeff(product,use)
    identity_matrix= np.identity(616)
    leontief_indus_technological_assump = pd.DataFrame(np.linalg.inv(identity_matrix - technical_coeff), index= technical_coeff.index, columns= technical_coeff.columns)
    indirect_consumption_indus_tech_assump = leontief_indus_technological_assump - identity_matrix - technical_coeff
    return leontief_indus_technological_assump, indirect_consumption_indus_tech_assump, technical_coeff



def leontief_commodity_tech_assumption(make_table_naf, use_table_naf):
    technical_coeff = calculate_tech_coeff_comm(make_table_naf,use_table_naf)
    identity_matrix= np.identity(616)
    leontief_commodity_technological_assump = pd.DataFrame(np.linalg.inv(identity_matrix - technical_coeff), index= technical_coeff.index, columns= technical_coeff.columns)
    indirect_consumption_comm_tech_assump = leontief_commodity_technological_assump - identity_matrix
    return leontief_commodity_technological_assump, indirect_consumption_comm_tech_assump 


def transition_nomenclature_nace(path):
    # On importe la feuille de la base de données avec les correspondanes produits et NAICS
    io_usa_trans = pd.read_excel(path+"Supply.ods", sheet_name="NAICS Codes", skiprows=4, usecols=[3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],engine="odf")
    # On supprime les lignes inutiles
    io_usa_trans = io_usa_trans.iloc[6:]
    # On renomme les colonnes pour les manipuler plus facilement
    #io_usa_trans.rename(columns={"Unnamed: 4":"Description", "Related 2012 NAICS Codes":"NAICS_2012"}, inplace=True)
    io_usa_trans = io_usa_trans.set_index("Detail")
    return io_usa_trans

def liste_nace(path):
    list_nace = pd.read_excel(path +"NACE_REV2-US_NAICS_2012.xls")
    list_nace =list_nace["NACE Rev. 2"].unique()
    list_nace = list_nace.astype("str")
    list_naces = [element for element in list_nace if len(element)>3 ]
    list_naces = ["0"+element  if len(element)<=4 and element[0]=="1"  else element for element in list_nace]
    return list_naces

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    path_data = "/Desktop/Matrice_Input_Output/fichier_i_o/"
    path_transition= "./Desktop/Matrice_Input_Output/transition_nomenclature/"
    use_table, transition_table, make_table, output_sector, tax_use  = importation_table(path_data, path_transition)
    use_table = preprocess_use_table(use_table)
    transition_table = preprocess_transition_table(transition_table)
    make_table = preprocess_make_table(make_table)
    make_table_naf = transition_make_bea_naf(make_table,transition_table)
    
    use_table_naf = transition_use_bea_naf(use_table, transition_table)
    output_use_naf = preprocess_output_vector(output_sector, transition_table)
    #tax_naf = transition_tax(tax_use, transition_table)
    product_coeff_mat = coeff_output(make_table_naf)
    intermediate_consumption = coeff_intermediate_consumption(use_table_naf,output_use_naf)
    leontief_indus_tech_assumption, indirect_consumption, technical_coeff = leontief_industrial_tech_assumption(product_coeff_mat, intermediate_consumption)
    leontief_commodity_technological_assump, indirect_consumption_comm_tech_assump = leontief_commodity_tech_assumption(make_table_naf,use_table_naf)

    use_table_naf.to_excel("./result/use.xlsx")
    make_table_naf.to_excel("./result/make.xlsx")
    output_use_naf.to_excel("./result/output_use_naf.xlsx")
    product_coeff_mat.to_excel("./result/product_coef.xlsx")
    intermediate_consumption.to_excel("./result/intermediate_consumption.xlsx")
    leontief_indus_tech_assumption.to_csv("./result/leontief_indus_tech_assumption.csv")
    indirect_consumption.to_csv("./result/indirect_consumption_indus_tech_assump.csv")
    leontief_commodity_technological_assump.to_excel("./result/leontief_commodity_technological_assump.xlsx")
    indirect_consumption_comm_tech_assump.to_excel("./result/indirect_consumption_comm_tech_assump.xlsx")
    technical_coeff.to_excel("./result/input_output_coefficient.xlsx")
    


main()

