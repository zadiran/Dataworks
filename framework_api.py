from measurement.utils import calculate_measurements_for_points

from models.point_forecast_model import point_forecast_model
from models.linear_regression_point_forecast_model import linear_regression_point_forecast_model

from data_processing import csv_data_source
from data_processing.configurable import configurable_data_manipulator
from data_processing.configurable.stages.pre_point_conversion import drop_columns
from data_processing.configurable.stages.point_conversion import convert_to_1d_input

from measurement.absolute import mean_absolute_error as mae


class training_result(object):
    dataset_id: str = None
    task: str = None
    class_machine: str = None
    requested_precision: float = None
    achieved_precision: float = None
    model: point_forecast_model  = None

    def __init__(self, dataset_id: str, task: str, class_machine: str, requested_precision: float, achieved_precision: float, model: point_forecast_model):
        self.dataset_id = dataset_id
        self.task = task
        self.class_machine = class_machine
        self.requested_precision = requested_precision
        self.achieved_precision = achieved_precision
        self.model = model

class forecasting_result(object):
    result: any = None
    achieved_precision: float = None

    def __init__(self, result: any, achieved_precision: float):
        self.result = result
        self.achieved_precision = achieved_precision


def get_data():
    
    window_size = 10

    raw_data = csv_data_source().get_data('data/train_FD001.csv', ';')

    cdm = configurable_data_manipulator('.local/cache/all_input_continuous.pickle')

    cdm.add_pre_point_conversion_stage(drop_columns(['s3', 's4', 's8', 's9', 's13', 's19', 's21', 's22']))
    cdm.set_point_conversion_stage(convert_to_1d_input(window_size))

    data = cdm.get_processed_data(raw_data)
    return data


def train_model(dataset_id: str, task: str, precision: float, class_machine: str):
    data = get_data()
    training_set = [x for x in data if x.unit <= 80]
    verification_set = [x for x in data if x.unit > 80 and x.unit <= 95]

    model = linear_regression_point_forecast_model()
    model.fit(training_set)
    model.predict_points(verification_set)

    measurements = calculate_measurements_for_points([mae()], verification_set)
 
    return training_result(dataset_id, task, class_machine, precision, measurements[0].value, model)


def forecast(model: point_forecast_model):
    data = get_data()
    forecasting_set = [x for x in data if x.unit > 95]

    model.predict_points(forecasting_set)

    measurements = calculate_measurements_for_points([mae()], forecasting_set)
    result = [x.forecasted_output for x in forecasting_set]
    return forecasting_result(result, measurements[0].value)
