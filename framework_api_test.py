from framework_api import train_model, forecast

result = train_model(1,2,3,4)
print(result.achieved_precision)

forecasting_result = forecast(result.model)
print(forecasting_result.achieved_precision)