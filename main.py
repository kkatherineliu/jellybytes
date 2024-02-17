import config
import cohere
import guardrails as gd
from guardrails.validators import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List
from flask import request, jsonify, Flask

# Flask for endpoints
app = Flask(__name__)

# For Pydantic Schema
class Sunscreen(BaseModel):
    name: str = Field(..., description="Name of the sunscreen")
    spf: str = Field(
        ..., 
        description="What is the SPF of the sunscreen?", 
        validators=[ValidChoices(["15", "30", "45", "50"], on_fail="reask")]
    )
    explanations: str = Field(..., description="Why does the sunscreen work well for patient's skin type and location?")

# Flask endpoint to produce a sunscreen recommendation
@app.route('/recommend-sunscreen', methods=['POST'])
def handle_reccomend_sunscreen():
    try:
        skin_type = request.json.get('skin_type')
        complexion = request.json.get('complexion')
        location = request.json.get('location')

        response = recommend_sunscreen(skin_type, complexion, location)

        return jsonify({'recommendation': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# calling the Cohere /chat endpoint to generate a response
def recommend_sunscreen(complexion, skin_type, location):

    # setting up the connection to Cohere API
    co = cohere.Client(config.api_key)

    response = co.chat(
        message = f'''You are a dermatologist. Given the following notes about a user,
        please reccomend a sunscreen that suits their skin and location needs and provide the
        sunscreen name, spf, and an explanation justifying the recommended choice.
        
        Fitzpatrick Skin Type: {complexion}
        Skin Type: {skin_type}
        Location: {location}''' + 

        ''' Please produce the reccomendation as a dictionary.
        Given below is XML that describes the information to extract from this document and the tags to extract it into.

        <output>
            <string name="name" description="Name of the sunscreen"/>
            <string name="spf" description="What is the SPF of the sunscreen?" format="valid-choices: choices=['15', '30', '45', 
        '50']"/>
            <string name="explanations" description="Why does the sunscreen work well for patient's skin type and location?"/>
        </output>


        ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute 
        of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to
        the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct 
        and concise.

        Here are examples of simple (XML, JSON) pairs that show the expected behavior:
        - `<string name='foo' format='two-words lower-case' />` => `{'foo': 'example one'}`
        - `<list name='bar'><string format='upper-case' /></list>` => `{"bar": ['STRING ONE', 'STRING TWO', etc.]}`
        - `<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" 
        /></object>` => `{'baz': {'foo': 'Some String', 'index': 1}}`''',

	    model="command",
        connectors=[{"id": "web-search"}], # can make custom connectors with sunscreen-relevant information
        prompt_truncation="AUTO",
	    temperature=0.2 # can change, higher = more randomness
    )

    print(response.text)
    
########### guardrails stuff??
# PROMPT = """Given the following instructions,
# please extract a dictionary that contains the sunscreens's information.

# ${recommendation}
# ${gr.complete_json_suffix_v2}
# """

# # using Guardrails AI for structured output, similar to Pydantic
# guard = gd.Guard.from_pydantic(Sunscreen, prompt=PROMPT)
# print(guard.base_prompt)

# raw_llm_output, validated_ouput = guard(
#     co.chat,
#     prompt_params={"recommendation": recommendation},
#     model='command',
#     connectors=[{"id": "web-search"}], # can make custom connectors later
#     prompt_truncation="AUTO",
#     temperature=0.2 # higher = more random
# )
# # return(validated_ouput)
# print(validated_ouput)

    # to look at the validation behind the scenes?
    # print(guard.state.most_recent_call.history[0].rich_group)
    # guard.guard_state.most_recent_call.tree


# def calculate_interval(region, spf):
# connect/relate to another API to find the UV index and compare with spf index
# to calculate how often to send a notification through flutter!

### for testing with output structure ###
recommend_sunscreen("Type 1", "dry", "San Francisco")
# print(recommend_sunscreen("Type 3", "oily", "Toronto"))