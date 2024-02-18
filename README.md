# Sunshine Guardian
ElleHacks 2023 - Summer Challenge ☀️

Check out our [DevPost](https://devpost.com/software/sunshine-guardian) for a video demo and a more detailed description of the process

[Repo for frontend (React Native/Expo)](https://github.com/kkatherineliu/sunshine-guardian-frontend)

## Notes on current status:
1) Running the Expo app opens on the Loading screen to test/show that the app and Flask endpoints are interacting properly to generate a sunscreen
2) Individually testing main.py shows accurate API responses from the backend
3) The design + prototyping in Figma is fully complete
4) Page navigation and text input works for the Expo app
5) The styling is very off for some pages of the app which is why input could not be gathered according to plan but the process would be similar if not easier than generating a sunscreen recommendation so I believe it would be relatively straightforward to implement the other functions (in terms of how the frontend and backend interact)

## Technologies used:
- Figma
- React Native/Expo
- Python (Flask, Pydantic, Guardrails AI)
- Cohere Chat API
- OpenWeatherMap API
- OpenUV API

## Setup/Installation Guidelines:
(I'm not sure how to set up requirements.txt to configure things as it runs but the following information may be helpful if you want to run the app yourself)

For main.py
- set up python virtual environment (in VS Code)
- `pip install cohere` to access cohere (note that you need a unique api key)
- `pip install guardrails-ai` for the output validation 
- `pip install pydantic` for the JSON Schema definition
- `pip install flask` for endpoits
- `pip install flask-cors` for CORS
- `export FLASK-APP="main.py"`
- `flask run` to start the server

To set up + run Expo App
- `npm install` for Node.js
- `npm install expo@latest` for Expo
- `npx expo install --fix` to update/sync necessary dependencies
- `npx expo install react-native-web react-dom @expo/metro-runtime` to enable running in the browser
- `npx expo start` to start the app
  
Next steps
- finish up the mobile app functionalities
- user authentication (Auth0)
- deploying the server (heroku or AWS)
