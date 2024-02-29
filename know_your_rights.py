from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token


legal_data = [
    {"keywords": ["name", "territory", "Union"], "issue": "Name and territory of the Union", "solution": "India, that is Bharat, shall be a Union of States."},
    {"keywords": ["admission", "establishment", "new States"], "issue": "Admission or establishment of new States", "solution": "Parliament may by law admit into the Union, or establish, new States on such terms and conditions as it thinks fit."},
    {"keywords": ["formation", "new States", "alteration", "areas", "boundaries", "names", "existing States"],
     "issue": "Formation of new States and alteration of areas, boundaries or names of existing States",
     "solution": "Parliament may by law form a new State by separation of territory from any State or by uniting two or more States or parts of States or by uniting any territory to a part of any State; increase or diminish the area of any State; alter the boundaries or name of any State."},
    {"keywords": ["laws", "made", "articles 2", "3", "amendment", "First", "Fourth Schedules", "supplemental", "incidental", "consequential matters"],
     "issue": "Laws made under articles 2 and 3 to provide for the amendment of the First and the Fourth Schedules and supplemental, incidental and consequential matters",
     "solution": "Any law referred to in article 2 or article 3 shall contain provisions for the amendment of the First Schedule and the Fourth Schedule as necessary, along with supplemental, incidental, and consequential provisions."},
    {"keywords": ["citizenship", "commencement", "Constitution"],
     "issue": "Citizenship at the commencement of the Constitution",
     "solution": "Every person who has his domicile in the territory of India and who was born in the territory of India, or whose parents were born in the territory of India, or who has been ordinarily resident in the territory of India for not less than five years immediately preceding such commencement, shall be a citizen of India."},
    {"keywords": ["economically weaker sections", "family income", "indicators", "economic disadvantage"],
 "issue": "Definition of economically weaker sections",
 "solutions": [
     "For the purposes of this article and article 16, 'economically weaker sections' shall be such as may be notified by the State from time to time on the basis of family income and other indicators of economic disadvantage."
 ]
},

    
    {"keywords": ["prevention", "detention", "preventive detention", "Advisory Board", "High Court"],
     "issue": "Protection against arrest and detention in certain cases",
     "solutions": [
         "(4) No law providing for preventive detention shall authorize the detention of a person for a longer period than three months unless an Advisory Board consisting of persons who are, or have been, or are qualified to be appointed as, Judges of a High Court has reported before the expiration of the said period of three months that there is, in its opinion, sufficient cause for such detention."
     ]
    }
]


def generate_legal_solution(input_text):
    
    input_text_lower = input_text.lower()

  
    matching_entries = [item for item in legal_data if any(keyword.lower() in input_text_lower for keyword in item["keywords"])]

    if matching_entries:
       
        dataset_entry = matching_entries[0]

     
        output = model.generate(tokenizer.encode(dataset_entry["issue"], return_tensors="pt", padding=True, truncation=True),
                                max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_solution = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_solution
    else:
        return "No legal solution found for the provided input."


input_text = ""
legal_solution = generate_legal_solution(input_text)

print("Input Text:", input_text)
print("Generated Legal Solution:", legal_solution)
