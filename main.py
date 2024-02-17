import config
import cohere
from flask import request, jsonify, Flask

# setting up the connection to Cohere API
co = cohere.Client(config.api_key)

# Flask for endpoints
app = Flask(__name__)

# Flask endpoint to produce a sunscreen recommendation
@app.route('/recommend-sunscreen', methods=['POST'])
def handle_reccomend_sunscreen():
    try:
        skin_type = request.json.get('skin_type')
        complexion = request.json.get('complexion')
        region = request.json.get('region')
        sunlight_exposure = request.json.get('sunlight_exposure')

        response = recommend_sunscreen(skin_type, complexion, region, sunlight_exposure)

        return jsonify({'recommendation': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# calling the Cohere /chat endpoint to generate a response
def recommend_sunscreen(skin_type, complexion, region, sunlight_exposure):
    response = co.chat(
        # if skin_type has multi-select, might be a list so you might have to parse
        # instead of creating a super long prompt, look into structured LLM outputs (pydantic? guardrails?)
        # because I don't think all the additional specificity is improving the format that much
        # 
        message = f'''You are a dermatologist, please recommend one singular sunscreen for me. I have 
        {skin_type} {complexion} skin who lives in {region} and usually spends around {sunlight_exposure} hours outside under the sun. 
        Indicate the name and SPF of the sunscreen first, then a 1-2 short sentences explaining why it is a good fit. 
        Please do not include any additional sentences.

        Follow this formatting: [Sunscreen Name] [SPF] - [Explanation]''',
	    model="command",
        connectors=[{"id": "web-search"}], # can make custom connectors with sunscreen-relevant information
        prompt_truncation="AUTO",
	    temperature=0.2 # can change, higher = more randomness
    )
    return response.text


# def calculate_interval(region, spf):
# connect/relate to another API to find the UV index and compare with spf index
# to calculate how often to send a notification through flutter!

### for testing with output structure ###
# print(recommend_sunscreen("oily", "tanned", "San Francisco", "7"))
# print(recommend_sunscreen("senstive dry", "dark brown", "Toronto", "2"))
# print(recommend_sunscreen("acne-pronce", "pale", "Mexico", "5"))


## previously part of the prompt but didn't seem to properly structure the response enough of the time
'''
    Here are a few examples of correctly formatted responses:
        
    'EltaMD Daily Tinted SPF 40 - This sunscreen includes zinc oxide, 
    a natural mineral compound that blocks a wide spectrum of UVA and UVB rays, 
    protecting the skin from damage and aging caused by the sun. 
    This ingredient is especially beneficial for dry skin as it maintains its protective ability in the sun, 
    reducing the risk of sunburn.'

    'SaieSunvisor Radiant Moisturizing Face Sunscreen SPF 35 - 
    This sunscreen is a great fit for you as it is specifically recommended 
    for dry skin and leaves a dewy finish on the skin. 
    It currently ranks at SPF 35, which is ideal for pale skin types.'

    'La Roche-Posay Anthelios Clear Skin Oil-Free SPF 60 - This sunscreen not only has an SPF rating of 60, 
    it is also formulated with silica and perlite to reduce oil production during the day. 
    It is also water-resistant, great for active users, and suitable for sensitive skin.

'''