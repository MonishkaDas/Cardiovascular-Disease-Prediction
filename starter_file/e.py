import requests
import json

          
# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://731d3ebf-3734-4213-9182-e56b83b97293.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'NfJhab6U5UDG3K1s8ir4IFV7lmofte8q'

# Two sets of data to score, so we get two results back
data = {"data": 
            [
                {
                    "age": -0.436058,
                    "gender": 0,
                    "height": 0.443449,
                    "weight": -0.847867,
                    "ap_hi": -0.122181, 
                    "ap_lo": -0.088238,
                    "smoke": 0,
                    "alco": 0,
                    "active": 1,
                    "cardio": 0,
                    "cholesterol_normal": 1,
                    "cholesterol_well above normal": 0,
                    "gluc_normal": 1,
                    "gluc_well above normal": 0
                 
                }, 
                {
                    "age": 0.307684,
                    "gender": 1,
                    "height": -1.018161,
                    "weight": 0.749826,
                    "ap_hi": 0.072610, 
                    "ap_lo": -0.035180,
                    "smoke": 0,
                    "alco": 0,
                    "active": 1,
                    "cardio": 1,
                    "cholesterol_normal": 0,
                    "cholesterol_well above normal": 1,
                    "gluc_normal": 1,
                    "gluc_well above normal": 0
                }, 
                {
                    "age": -0.247995,
                    "gender": 1,
                    "height": 0.078046,
                    "weight": -0.708937,
                    "ap_hi": 0.007679, 
                    "ap_lo": -0.141296,
                    "smoke": 0,
                    "alco": 0,
                    "active": 0,
                    "cardio": 1,
                    "cholesterol_normal": 0,
                    "cholesterol_well above normal": 1,
                    "gluc_normal": 1,
                    "gluc_well above normal": 0
                }
            ]
        }

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
