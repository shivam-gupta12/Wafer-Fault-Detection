import os
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from logger import logging
from exception import CustomException
from components import data_ingestion
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from utils import save_object
from imblearn.combine import SMOTETomek
from sklearn.compose import ColumnTransformer


@dataclass
class datatransformationconfig:
    preprocessor_path = os.path.join('artifacts' , 'preprocessor.pkl')
    
class DataTransformation():
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("establishing pipeline")
            numerical_cols = ['Sensor-1', 'Sensor-2', 'Sensor-3', 'Sensor-4', 'Sensor-5', 'Sensor-7', 'Sensor-8', 'Sensor-9', 'Sensor-10', 'Sensor-11', 'Sensor-12', 'Sensor-13', 'Sensor-15', 'Sensor-16', 'Sensor-17', 'Sensor-18', 'Sensor-19', 'Sensor-20','Sensor-21', 'Sensor-22', 'Sensor-23', 'Sensor-24', 'Sensor-25', 'Sensor-26', 'Sensor-27', 'Sensor-28', 'Sensor-29','Sensor-30', 'Sensor-31', 'Sensor-32', 'Sensor-33', 'Sensor-34', 'Sensor-35', 'Sensor-36', 'Sensor-37', 'Sensor-38','Sensor-39', 'Sensor-40', 'Sensor-41', 'Sensor-42', 'Sensor-44', 'Sensor-45', 'Sensor-46', 'Sensor-47', 'Sensor-48','Sensor-49', 'Sensor-51', 'Sensor-52', 'Sensor-54', 'Sensor-55', 'Sensor-56', 'Sensor-57', 'Sensor-58', 'Sensor-59','Sensor-60', 'Sensor-61', 'Sensor-62', 'Sensor-63', 'Sensor-64', 'Sensor-65', 'Sensor-66', 'Sensor-67', 'Sensor-68','Sensor-69', 'Sensor-71', 'Sensor-72', 'Sensor-73', 'Sensor-74', 'Sensor-76', 'Sensor-77', 'Sensor-78', 'Sensor-79','Sensor-80', 'Sensor-81', 'Sensor-82', 'Sensor-83', 'Sensor-84', 'Sensor-85', 'Sensor-86', 'Sensor-87', 'Sensor-88','Sensor-89', 'Sensor-90', 'Sensor-91', 'Sensor-92', 'Sensor-93', 'Sensor-94', 'Sensor-95', 'Sensor-96', 'Sensor-97','Sensor-99', 'Sensor-100', 'Sensor-101', 'Sensor-102', 'Sensor-103', 'Sensor-104', 'Sensor-105', 'Sensor-106','Sensor-107', 'Sensor-108', 'Sensor-109', 'Sensor-110', 'Sensor-111', 'Sensor-112', 'Sensor-113', 'Sensor-114','Sensor-115', 'Sensor-116', 'Sensor-117', 'Sensor-118', 'Sensor-119', 'Sensor-120', 'Sensor-121', 'Sensor-122','Sensor-123', 'Sensor-124', 'Sensor-125', 'Sensor-126', 'Sensor-127', 'Sensor-128', 'Sensor-129', 'Sensor-130','Sensor-131', 'Sensor-132', 'Sensor-133', 'Sensor-134', 'Sensor-135', 'Sensor-136', 'Sensor-137', 'Sensor-138','Sensor-139', 'Sensor-140', 'Sensor-141', 'Sensor-143', 'Sensor-144', 'Sensor-145', 'Sensor-146', 'Sensor-147','Sensor-148', 'Sensor-149', 'Sensor-151', 'Sensor-152', 'Sensor-153', 'Sensor-154', 'Sensor-155', 'Sensor-156','Sensor-157', 'Sensor-160', 'Sensor-161', 'Sensor-162', 'Sensor-163', 'Sensor-164', 'Sensor-165', 'Sensor-166','Sensor-167', 'Sensor-168', 'Sensor-169', 'Sensor-170', 'Sensor-171', 'Sensor-172', 'Sensor-173', 'Sensor-174','Sensor-175', 'Sensor-176', 'Sensor-177', 'Sensor-178', 'Sensor-181', 'Sensor-182', 'Sensor-183', 'Sensor-184','Sensor-185', 'Sensor-186', 'Sensor-188', 'Sensor-189', 'Sensor-196', 'Sensor-197', 'Sensor-198', 'Sensor-199','Sensor-200', 'Sensor-201', 'Sensor-202', 'Sensor-203', 'Sensor-204', 'Sensor-205', 'Sensor-206', 'Sensor-208','Sensor-209', 'Sensor-211', 'Sensor-212', 'Sensor-213', 'Sensor-214', 'Sensor-215', 'Sensor-216', 'Sensor-217','Sensor-218', 'Sensor-219', 'Sensor-220', 'Sensor-221', 'Sensor-222', 'Sensor-223', 'Sensor-224', 'Sensor-225','Sensor-226', 'Sensor-228', 'Sensor-229', 'Sensor-239', 'Sensor-240', 'Sensor-245', 'Sensor-246', 'Sensor-247','Sensor-248', 'Sensor-249', 'Sensor-250', 'Sensor-251', 'Sensor-252', 'Sensor-253', 'Sensor-254', 'Sensor-255','Sensor-256', 'Sensor-268', 'Sensor-269', 'Sensor-270', 'Sensor-271', 'Sensor-272', 'Sensor-273', 'Sensor-274','Sensor-275', 'Sensor-276', 'Sensor-278', 'Sensor-279', 'Sensor-280', 'Sensor-281', 'Sensor-282', 'Sensor-283','Sensor-284', 'Sensor-286', 'Sensor-287', 'Sensor-288', 'Sensor-289', 'Sensor-290', 'Sensor-291', 'Sensor-292','Sensor-295', 'Sensor-296', 'Sensor-297', 'Sensor-298', 'Sensor-299', 'Sensor-300', 'Sensor-301', 'Sensor-302','Sensor-303', 'Sensor-304', 'Sensor-305', 'Sensor-306', 'Sensor-307', 'Sensor-308', 'Sensor-309', 'Sensor-310','Sensor-311', 'Sensor-312', 'Sensor-313', 'Sensor-317', 'Sensor-318', 'Sensor-319', 'Sensor-320', 'Sensor-321','Sensor-322', 'Sensor-324', 'Sensor-325', 'Sensor-332', 'Sensor-333', 'Sensor-334', 'Sensor-335', 'Sensor-336','Sensor-337', 'Sensor-338', 'Sensor-339', 'Sensor-340', 'Sensor-341', 'Sensor-342', 'Sensor-344', 'Sensor-345','Sensor-346', 'Sensor-347', 'Sensor-349', 'Sensor-350', 'Sensor-351', 'Sensor-352', 'Sensor-353', 'Sensor-354','Sensor-355', 'Sensor-356', 'Sensor-357', 'Sensor-358', 'Sensor-359', 'Sensor-360', 'Sensor-361', 'Sensor-362','Sensor-363', 'Sensor-364', 'Sensor-366', 'Sensor-367', 'Sensor-368', 'Sensor-369', 'Sensor-377', 'Sensor-378','Sensor-383', 'Sensor-384', 'Sensor-385', 'Sensor-386', 'Sensor-387', 'Sensor-388', 'Sensor-389', 'Sensor-390','Sensor-391', 'Sensor-392', 'Sensor-393', 'Sensor-394', 'Sensor-406', 'Sensor-407', 'Sensor-408', 'Sensor-409','Sensor-410', 'Sensor-411', 'Sensor-412', 'Sensor-413', 'Sensor-414', 'Sensor-416', 'Sensor-417', 'Sensor-418','Sensor-419', 'Sensor-420', 'Sensor-421', 'Sensor-422', 'Sensor-424', 'Sensor-425', 'Sensor-426', 'Sensor-427','Sensor-428', 'Sensor-429', 'Sensor-430', 'Sensor-431', 'Sensor-432', 'Sensor-433', 'Sensor-434', 'Sensor-435','Sensor-436', 'Sensor-437', 'Sensor-438', 'Sensor-439', 'Sensor-440', 'Sensor-441', 'Sensor-442', 'Sensor-443','Sensor-444', 'Sensor-445', 'Sensor-446', 'Sensor-447', 'Sensor-448', 'Sensor-449', 'Sensor-453', 'Sensor-454','Sensor-455', 'Sensor-456', 'Sensor-457', 'Sensor-458', 'Sensor-460', 'Sensor-461', 'Sensor-468', 'Sensor-469','Sensor-470', 'Sensor-471', 'Sensor-472', 'Sensor-473', 'Sensor-474', 'Sensor-475', 'Sensor-476', 'Sensor-477','Sensor-478', 'Sensor-480', 'Sensor-481', 'Sensor-483', 'Sensor-484', 'Sensor-485', 'Sensor-486', 'Sensor-487','Sensor-488', 'Sensor-489', 'Sensor-490', 'Sensor-491', 'Sensor-492', 'Sensor-493', 'Sensor-494', 'Sensor-495','Sensor-496', 'Sensor-497', 'Sensor-498', 'Sensor-500', 'Sensor-501', 'Sensor-511', 'Sensor-512', 'Sensor-517','Sensor-518', 'Sensor-519', 'Sensor-520', 'Sensor-521', 'Sensor-522', 'Sensor-523', 'Sensor-524', 'Sensor-525','Sensor-526', 'Sensor-527', 'Sensor-528', 'Sensor-540', 'Sensor-541', 'Sensor-542', 'Sensor-543', 'Sensor-544','Sensor-545', 'Sensor-546', 'Sensor-547', 'Sensor-548', 'Sensor-549', 'Sensor-550', 'Sensor-551', 'Sensor-552','Sensor-553', 'Sensor-554', 'Sensor-555', 'Sensor-556', 'Sensor-557', 'Sensor-558', 'Sensor-559', 'Sensor-560','Sensor-561', 'Sensor-562', 'Sensor-563', 'Sensor-564', 'Sensor-565', 'Sensor-566', 'Sensor-567', 'Sensor-568','Sensor-569', 'Sensor-570', 'Sensor-571', 'Sensor-572', 'Sensor-573', 'Sensor-574', 'Sensor-575', 'Sensor-576','Sensor-577', 'Sensor-578', 'Sensor-579', 'Sensor-580', 'Sensor-581', 'Sensor-582', 'Sensor-583', 'Sensor-584','Sensor-585', 'Sensor-586', 'Sensor-587', 'Sensor-588', 'Sensor-589', 'Sensor-590']
            imputer = KNNImputer(n_neighbors=3)
            preprocessing_pipeline = Pipeline(
                steps = [
                    ('imputer' , imputer),
                    ('scalar' , RobustScaler()),
                    #('resampler' , SMOTETomek(sampling_strategy="auto"))
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('pipeline' , preprocessing_pipeline , numerical_cols)
            ])
            
            logging.info("transformation pipeline built")
            return preprocessor
        
        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info("data transformation started")

            train_df = pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)
            
            X_train = train_df.drop('Good/Bad', axis=1)
            y_train = train_df['Good/Bad']

            X_test = test_df.drop('Good/Bad', axis=1)
            y_test = test_df['Good/Bad']
            
            print(X_train)
            preprocessor = self.get_data_transformation_object()
            df_trans_train = preprocessor.fit_transform(X_train)
            df_trans_test = preprocessor.transform(X_test)
            
            train_arr = np.c_[df_trans_train , np.array(y_train)]
            test_arr = np.c_[df_trans_test , np.array(y_test)]
            
            logging.info("data transformation completed")
            save_object(file_path='artifacts/preprocessor.pkl' , obj=preprocessor)
            logging.info("saved preprocessor.pkl")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
        
        
        