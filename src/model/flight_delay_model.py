import pandas as pd
import joblib
import xgboost as xgb
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset
import time
import streamlit as st
from typing import Dict


# Model
class FlightDelayModel:
    """
    The Model component for the Flight Delay Prediction App.

    This class handles data loading, model loading, and delay predictions.
    """

    def __init__(self, model_file="models/best_model.pkl"):
        """
        Initializes the FlightDelayModel.

        Args:
            model_file (str): Path to the pre-trained machine learning model.
        """
        self.model = joblib.load(model_file)
        self.target = 'ArrDelay'            
        self.columns_for_df = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'FlightNum','CRSElapsedTime', 'AirTime', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'UniqueCarrier_AA', 'UniqueCarrier_AQ', 'UniqueCarrier_AS', 'UniqueCarrier_B6', 'UniqueCarrier_CO', 'UniqueCarrier_DL', 'UniqueCarrier_EV', 'UniqueCarrier_F9', 'UniqueCarrier_FL', 'UniqueCarrier_HA', 'UniqueCarrier_MQ', 'UniqueCarrier_NW', 'UniqueCarrier_OH', 'UniqueCarrier_OO', 'UniqueCarrier_UA', 'UniqueCarrier_US', 'UniqueCarrier_WN', 'UniqueCarrier_XE', 'UniqueCarrier_YV', 'Origin_ABI', 'Origin_ABQ', 'Origin_ABY', 'Origin_ACK', 'Origin_ACT', 'Origin_ACV', 'Origin_ACY', 'Origin_ADK', 'Origin_ADQ', 'Origin_AEX', 'Origin_AGS', 'Origin_AKN', 'Origin_ALB', 'Origin_ALO', 'Origin_AMA', 'Origin_ANC', 'Origin_ASE', 'Origin_ATL', 'Origin_ATW', 'Origin_AUS', 'Origin_AVL', 'Origin_AVP', 'Origin_AZO', 'Origin_BDL', 'Origin_BET', 'Origin_BFL', 'Origin_BGM', 'Origin_BGR', 'Origin_BHM', 'Origin_BIL', 'Origin_BIS', 'Origin_BJI', 'Origin_BLI', 'Origin_BMI', 'Origin_BNA', 'Origin_BOI', 'Origin_BOS', 'Origin_BPT', 'Origin_BQK', 'Origin_BQN', 'Origin_BRO', 'Origin_BRW', 'Origin_BTM', 'Origin_BTR', 'Origin_BTV', 'Origin_BUF', 'Origin_BUR', 'Origin_BWI', 'Origin_BZN', 'Origin_CAE', 'Origin_CAK', 'Origin_CDC', 'Origin_CDV', 'Origin_CEC', 'Origin_CHA', 'Origin_CHO', 'Origin_CHS', 'Origin_CIC', 'Origin_CID', 'Origin_CLD', 'Origin_CLE', 'Origin_CLL', 'Origin_CLT', 'Origin_CMH', 'Origin_CMI', 'Origin_CMX', 'Origin_COD', 'Origin_COS', 'Origin_CPR', 'Origin_CRP', 'Origin_CRW', 'Origin_CSG', 'Origin_CVG', 'Origin_CWA', 'Origin_DAB', 'Origin_DAL', 'Origin_DAY', 'Origin_DBQ', 'Origin_DCA', 'Origin_DEN', 'Origin_DFW', 'Origin_DHN', 'Origin_DLG', 'Origin_DLH', 'Origin_DRO', 'Origin_DSM', 'Origin_DTW', 'Origin_EGE', 'Origin_EKO', 'Origin_ELM', 'Origin_ELP', 'Origin_ERI', 'Origin_EUG', 'Origin_EVV', 'Origin_EWN', 'Origin_EWR', 'Origin_EYW', 'Origin_FAI', 'Origin_FAR', 'Origin_FAT', 'Origin_FAY', 'Origin_FCA', 'Origin_FLG', 'Origin_FLL', 'Origin_FLO', 'Origin_FNT', 'Origin_FSD', 'Origin_FSM', 'Origin_FWA', 'Origin_GCC', 'Origin_GEG', 'Origin_GFK', 'Origin_GGG', 'Origin_GJT', 'Origin_GNV', 'Origin_GPT', 'Origin_GRB', 'Origin_GRK', 'Origin_GRR', 'Origin_GSO', 'Origin_GSP', 'Origin_GTF', 'Origin_GTR', 'Origin_GUC', 'Origin_HDN', 'Origin_HHH', 'Origin_HLN', 'Origin_HNL', 'Origin_HOU', 'Origin_HPN', 'Origin_HRL', 'Origin_HSV', 'Origin_IAD', 'Origin_IAH', 'Origin_ICT', 'Origin_IDA', 'Origin_ILM', 'Origin_IND', 'Origin_INL', 'Origin_IPL', 'Origin_ISP', 'Origin_ITO', 'Origin_IYK', 'Origin_JAC', 'Origin_JAN', 'Origin_JAX', 'Origin_JFK', 'Origin_JNU', 'Origin_KOA', 'Origin_KTN', 'Origin_LAN', 'Origin_LAS', 'Origin_LAW', 'Origin_LAX', 'Origin_LBB', 'Origin_LCH', 'Origin_LEX', 'Origin_LFT', 'Origin_LGA', 'Origin_LGB', 'Origin_LIH', 'Origin_LIT', 'Origin_LNK', 'Origin_LRD', 'Origin_LSE', 'Origin_LWB', 'Origin_LWS', 'Origin_LYH', 'Origin_MAF', 'Origin_MBS', 'Origin_MCI', 'Origin_MCN', 'Origin_MCO', 'Origin_MDT', 'Origin_MDW', 'Origin_MEI', 'Origin_MEM', 'Origin_MFE', 'Origin_MFR', 'Origin_MGM', 'Origin_MHT', 'Origin_MIA', 'Origin_MKE', 'Origin_MKG', 'Origin_MLB', 'Origin_MLI', 'Origin_MLU', 'Origin_MOB', 'Origin_MOD', 'Origin_MOT', 'Origin_MQT', 'Origin_MRY', 'Origin_MSN', 'Origin_MSO', 'Origin_MSP', 'Origin_MSY', 'Origin_MTJ', 'Origin_MYR', 'Origin_OAJ', 'Origin_OAK', 'Origin_OGG', 'Origin_OKC', 'Origin_OMA', 'Origin_OME', 'Origin_ONT', 'Origin_ORD', 'Origin_ORF', 'Origin_OTZ', 'Origin_OXR', 'Origin_PBI', 'Origin_PDX', 'Origin_PFN', 'Origin_PHF', 'Origin_PHL', 'Origin_PHX', 'Origin_PIA', 'Origin_PIH', 'Origin_PIT', 'Origin_PLN', 'Origin_PMD', 'Origin_PNS', 'Origin_PSC', 'Origin_PSE', 'Origin_PSG', 'Origin_PSP', 'Origin_PVD', 'Origin_PWM', 'Origin_RAP', 'Origin_RDD', 'Origin_RDM', 'Origin_RDU', 'Origin_RFD', 'Origin_RHI', 'Origin_RIC', 'Origin_RKS', 'Origin_RNO', 'Origin_ROA', 'Origin_ROC', 'Origin_ROW', 'Origin_RST', 'Origin_RSW', 'Origin_SAN', 'Origin_SAT', 'Origin_SAV', 'Origin_SBA', 'Origin_SBN', 'Origin_SBP', 'Origin_SCC', 'Origin_SCE', 'Origin_SDF', 'Origin_SEA', 'Origin_SFO', 'Origin_SGF', 'Origin_SGU', 'Origin_SHV', 'Origin_SIT', 'Origin_SJC', 'Origin_SJT', 'Origin_SJU', 'Origin_SLC', 'Origin_SLE', 'Origin_SMF', 'Origin_SMX', 'Origin_SNA', 'Origin_SPI', 'Origin_SPS', 'Origin_SRQ', 'Origin_STL', 'Origin_STT', 'Origin_STX', 'Origin_SUN', 'Origin_SUX', 'Origin_SWF', 'Origin_SYR', 'Origin_TEX', 'Origin_TLH', 'Origin_TOL', 'Origin_TPA', 'Origin_TRI', 'Origin_TUL', 'Origin_TUP', 'Origin_TUS', 'Origin_TVC', 'Origin_TWF', 'Origin_TXK', 'Origin_TYR', 'Origin_TYS', 'Origin_VLD', 'Origin_VPS', 'Origin_WRG', 'Origin_WYS', 'Origin_XNA', 'Origin_YAK', 'Origin_YKM', 'Origin_YUM', 'Dest_ABI', 'Dest_ABQ', 'Dest_ABY', 'Dest_ACK', 'Dest_ACT', 'Dest_ACV', 'Dest_ACY', 'Dest_ADK', 'Dest_ADQ', 'Dest_AEX', 'Dest_AGS', 'Dest_AKN', 'Dest_ALB', 'Dest_ALO', 'Dest_AMA', 'Dest_ANC', 'Dest_ASE', 'Dest_ATL', 'Dest_ATW', 'Dest_AUS', 'Dest_AVL', 'Dest_AVP', 'Dest_AZO', 'Dest_BDL', 'Dest_BET', 'Dest_BFL', 'Dest_BGM', 'Dest_BGR', 'Dest_BHM', 'Dest_BIL', 'Dest_BIS', 'Dest_BJI', 'Dest_BLI', 'Dest_BMI', 'Dest_BNA', 'Dest_BOI', 'Dest_BOS', 'Dest_BPT', 'Dest_BQK', 'Dest_BQN', 'Dest_BRO', 'Dest_BRW', 'Dest_BTM', 'Dest_BTR', 'Dest_BTV', 'Dest_BUF', 'Dest_BUR', 'Dest_BWI', 'Dest_BZN', 'Dest_CAE', 'Dest_CAK', 'Dest_CDC', 'Dest_CDV', 'Dest_CEC', 'Dest_CHA', 'Dest_CHO', 'Dest_CHS', 'Dest_CIC', 'Dest_CID', 'Dest_CLD', 'Dest_CLE', 'Dest_CLL', 'Dest_CLT', 'Dest_CMH', 'Dest_CMI', 'Dest_CMX', 'Dest_COD', 'Dest_COS', 'Dest_CPR', 'Dest_CRP', 'Dest_CRW', 'Dest_CSG', 'Dest_CVG', 'Dest_CWA', 'Dest_CYS', 'Dest_DAB', 'Dest_DAL', 'Dest_DAY', 'Dest_DBQ', 'Dest_DCA', 'Dest_DEN', 'Dest_DFW', 'Dest_DHN', 'Dest_DLG', 'Dest_DLH', 'Dest_DRO', 'Dest_DSM', 'Dest_DTW', 'Dest_EGE', 'Dest_EKO', 'Dest_ELM', 'Dest_ELP', 'Dest_ERI', 'Dest_EUG', 'Dest_EVV', 'Dest_EWN', 'Dest_EWR', 'Dest_EYW', 'Dest_FAI', 'Dest_FAR', 'Dest_FAT', 'Dest_FAY', 'Dest_FCA', 'Dest_FLG', 'Dest_FLL', 'Dest_FLO', 'Dest_FNT', 'Dest_FSD', 'Dest_FSM', 'Dest_FWA', 'Dest_GCC', 'Dest_GEG', 'Dest_GFK', 'Dest_GGG', 'Dest_GJT', 'Dest_GNV', 'Dest_GPT', 'Dest_GRB', 'Dest_GRK', 'Dest_GRR', 'Dest_GSO', 'Dest_GSP', 'Dest_GTF', 'Dest_GTR', 'Dest_GUC', 'Dest_HDN', 'Dest_HHH', 'Dest_HLN', 'Dest_HNL', 'Dest_HOU', 'Dest_HPN', 'Dest_HRL', 'Dest_HSV', 'Dest_IAD', 'Dest_IAH', 'Dest_ICT', 'Dest_IDA', 'Dest_ILM', 'Dest_IND', 'Dest_INL', 'Dest_IPL', 'Dest_ISP', 'Dest_ITO', 'Dest_IYK', 'Dest_JAC', 'Dest_JAN', 'Dest_JAX', 'Dest_JFK', 'Dest_JNU', 'Dest_KOA', 'Dest_KTN', 'Dest_LAN', 'Dest_LAS', 'Dest_LAW', 'Dest_LAX', 'Dest_LBB', 'Dest_LCH', 'Dest_LEX', 'Dest_LFT', 'Dest_LGA', 'Dest_LGB', 'Dest_LIH', 'Dest_LIT', 'Dest_LNK', 'Dest_LRD', 'Dest_LSE', 'Dest_LWB', 'Dest_LWS', 'Dest_LYH', 'Dest_MAF', 'Dest_MBS', 'Dest_MCI', 'Dest_MCN', 'Dest_MCO', 'Dest_MDT', 'Dest_MDW', 'Dest_MEI', 'Dest_MEM', 'Dest_MFE', 'Dest_MFR', 'Dest_MGM', 'Dest_MHT', 'Dest_MIA', 'Dest_MKE', 'Dest_MKG', 'Dest_MLB', 'Dest_MLI', 'Dest_MLU', 'Dest_MOB', 'Dest_MOD', 'Dest_MOT', 'Dest_MQT', 'Dest_MRY', 'Dest_MSN', 'Dest_MSO', 'Dest_MSP', 'Dest_MSY', 'Dest_MTJ', 'Dest_MYR', 'Dest_OAJ', 'Dest_OAK', 'Dest_OGD', 'Dest_OGG', 'Dest_OKC', 'Dest_OMA', 'Dest_OME', 'Dest_ONT', 'Dest_ORD', 'Dest_ORF', 'Dest_OTZ', 'Dest_OXR', 'Dest_PBI', 'Dest_PDX', 'Dest_PFN', 'Dest_PHF', 'Dest_PHL', 'Dest_PHX', 'Dest_PIA', 'Dest_PIH', 'Dest_PIT', 'Dest_PLN', 'Dest_PMD', 'Dest_PNS', 'Dest_PSC', 'Dest_PSE', 'Dest_PSG', 'Dest_PSP', 'Dest_PVD', 'Dest_PWM', 'Dest_RAP', 'Dest_RDD', 'Dest_RDM', 'Dest_RDU', 'Dest_RFD', 'Dest_RHI', 'Dest_RIC', 'Dest_RKS', 'Dest_RNO', 'Dest_ROA', 'Dest_ROC', 'Dest_ROW', 'Dest_RST', 'Dest_RSW', 'Dest_SAN', 'Dest_SAT', 'Dest_SAV', 'Dest_SBA', 'Dest_SBN', 'Dest_SBP', 'Dest_SCC', 'Dest_SCE', 'Dest_SDF', 'Dest_SEA', 'Dest_SFO', 'Dest_SGF', 'Dest_SGU', 'Dest_SHV', 'Dest_SIT', 'Dest_SJC', 'Dest_SJT', 'Dest_SJU', 'Dest_SLC', 'Dest_SLE', 'Dest_SMF', 'Dest_SMX', 'Dest_SNA', 'Dest_SPI', 'Dest_SPS', 'Dest_SRQ', 'Dest_STL', 'Dest_STT', 'Dest_STX', 'Dest_SUN', 'Dest_SUX', 'Dest_SWF', 'Dest_SYR', 'Dest_TEX', 'Dest_TLH', 'Dest_TOL', 'Dest_TPA', 'Dest_TRI', 'Dest_TUL', 'Dest_TUP', 'Dest_TUS', 'Dest_TVC', 'Dest_TWF', 'Dest_TXK', 'Dest_TYR', 'Dest_TYS', 'Dest_VLD', 'Dest_VPS', 'Dest_WRG', 'Dest_WYS', 'Dest_XNA', 'Dest_YAK', 'Dest_YKM', 'Dest_YUM']
        

        self.numerical_features = [
                'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'FlightNum',
                'CRSElapsedTime', 'AirTime', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
                'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
            ]
        
        self.categorical_options = {
            'UniqueCarrier': ['AA', 'AQ', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'NW', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV'],
            'Origin': ['ABI', 'ABQ', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALO', 'AMA', 'ANC', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO', 'CHS', 'CIC', 'CID', 'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'COD', 'COS', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB', 'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DLG', 'DLH', 'DRO', 'DSM', 'DTW', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTF', 'GTR', 'GUC', 'HDN', 'HHH', 'HLN', 'HNL', 'HOU', 'HPN', 'HRL', 'HSV', 'IAD', 'IAH', 'ICT', 'IDA', 'ILM', 'IND', 'INL', 'IPL', 'ISP', 'ITO', 'IYK', 'JAC', 'JAN', 'JAX', 'JFK', 'JNU', 'KOA', 'KTN', 'LAN', 'LAS', 'LAW', 'LAX', 'LBB', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF', 'MBS', 'MCI', 'MCN', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 'MGM', 'MHT', 'MIA', 'MKE', 'MKG', 'MLB', 'MLI', 'MLU', 'MOB', 'MOD', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'OTZ', 'OXR', 'PBI', 'PDX', 'PFN', 'PHF', 'PHL', 'PHX', 'PIA', 'PIH', 'PIT', 'PLN', 'PMD', 'PNS', 'PSC', 'PSE', 'PSG', 'PSP', 'PVD', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RFD', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF', 'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SMF', 'SNA', 'SPI', 'SPS', 'SRQ', 'STL', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SYR', 'TEX', 'TLH', 'TOL', 'TPA', 'TRI', 'TUL', 'TUP', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'VEL', 'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'YAK', 'YKM', 'YUM'],
            'Dest': ['ABI', 'ABQ', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALO', 'AMA', 'ANC', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO', 'CHS', 'CIC', 'CID', 'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'COD', 'COS', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB', 'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DLG', 'DLH', 'DRO', 'DSM', 'DTW', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTF', 'GTR', 'GUC', 'HDN', 'HHH', 'HLN', 'HNL', 'HOU', 'HPN', 'HRL', 'HSV', 'IAD', 'IAH', 'ICT', 'IDA', 'ILM', 'IND', 'INL', 'IPL', 'ISP', 'ITO', 'IYK', 'JAC', 'JAN', 'JAX', 'JFK', 'JNU', 'KOA', 'KTN', 'LAN', 'LAS', 'LAW', 'LAX', 'LBB', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF', 'MBS', 'MCI', 'MCN', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 'MGM', 'MHT', 'MIA', 'MKE', 'MKG', 'MLB', 'MLI', 'MLU', 'MOB', 'MOD', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'OTZ', 'OXR', 'PBI', 'PDX', 'PFN', 'PHF', 'PHL', 'PHX', 'PIA', 'PIH', 'PIT', 'PLN', 'PMD', 'PNS', 'PSC', 'PSE', 'PSG', 'PSP', 'PVD', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RFD', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF', 'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SLE', 'SMF', 'SMX', 'SNA', 'SPI', 'SPS', 'SRQ', 'STL', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SYR', 'TEX', 'TLH', 'TOL', 'TPA', 'TRI', 'TUL', 'TUP', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'YAK', 'YKM', 'YUM']
        }
    
        # Define column mapping
        self.column_mapping = ColumnMapping()
        self.column_mapping.target = self.target
        self.column_mapping.prediction = 'prediction'
        self.column_mapping.numerical_features = self.numerical_features

        
    def selected_data(self) -> Dict[str, int]:
        selected_data={col: 0 for col in self.columns_for_df}
        return selected_data

    def categorical_features(self) -> Dict[str, list]:
        return self.categorical_options    

    def predict_delay(self, input_data: pd.DataFrame) -> float:
        """
        Predicts flight delay based on user input.

        Args:
            input_data (pd.DataFrame): User-provided input data.

        Returns:
            float: Predicted flight delay in minutes.
        """
        return self.model.predict(input_data)

    def train_model(self,reference_data: pd.DataFrame, current_data: pd.DataFrame):
        # Create and train the XGBoost Regressor
        model=xgb.XGBRegressor()
        model_training_start_time = time.time()
        model.fit(reference_data[self.numerical_features], reference_data[self.target])
        ref_prediction = model.predict(reference_data[self.numerical_features])
        current_prediction = model.predict(current_data[self.numerical_features])
        model_training_end_time = time.time()
        st.write(f"Time taken for Model Training: {model_training_end_time - model_training_start_time} seconds")
        
        reference_data['prediction'] = ref_prediction
        current_data['prediction'] = current_prediction
        

    # Model performance report
    def performance_report(self,reference_data: pd.DataFrame, current_data: pd.DataFrame):
        regression_performance_report = Report(metrics=[RegressionPreset()])
        regression_performance_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return regression_performance_report

    
    def target_report(self,reference_data: pd.DataFrame, current_data: pd.DataFrame):
        target_drift_report = Report(metrics=[TargetDriftPreset()])
        target_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return target_drift_report
    
    
    def data_drift_report(self,reference_data: pd.DataFrame, current_data: pd.DataFrame):
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return data_drift_report
    
    
    def data_quality_report(self,reference_data: pd.DataFrame, current_data: pd.DataFrame):
        st.write("Generating the Data Quality Report will take more time, around 10 minutes, due to its thorough analysis. You can either wait or explore other reports if you're short on time.")
        data_quality_report = Report(metrics=[DataQualityPreset()])
        data_quality_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return data_quality_report