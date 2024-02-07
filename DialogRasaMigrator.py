"""
DIALOGFLOW TO RASA MIGRATOR

This script is designed to migrate Dialogflow chatbot data (intents and entities) to Rasa 3.
It processes JSON files exported from Dialogflow and generates corresponding YAML files for use in Rasa projects.
The script expects Dialogflow export files in JSON format and generates Rasa 3 compatible YAML files for intents,
entities, and training data.

Author: Natallia
Last Edited: 2024-02-03
Compatibility: Rasa 3.x
"""

import sys
import json
import os
import re

#-----------------------------------------------------------------------------------------------
# Global Configuration Variables
# These variables define the input and output directories for the migration process, 
# specifying where to find Dialogflow export files and where to place the generated Rasa files.
#-----------------------------------------------------------------------------------------------
DIALOGFLOW_INPUT_DIR    = 'dialogflow'          # Directory containing Dialogflow export files
DIALOGFLOW_ENTITIES_DIR = 'dialogflow/entities' # Subdirectory for Dialogflow entities
DIALOGFLOW_INTENTS_DIR  = 'dialogflow/intents'  # Subdirectory for Dialogflow intents
RASA_OUTPUT_DIR         = 'rasa'                # Main directory for generated Rasa files
RASA_DATA_DIR           = 'rasa/data'           # Directory for Rasa training data files
DEFAULT_GROUP           = 'OTHER'               # Default group for intents or entities not categorized by Dialogflow
IGNORE_SYSTEM_ENTITIES  = None                  # Flag to ignore system entities, set as needed
LANGUAGE_SETTING        = None                  # Language of the Dialogflow bot, to be set by the script

# Internal State Variables
# These lists and strings are used to store processed entities, intents, and rules 
# as the script converts Dialogflow data into the Rasa format.
ENTITY_LIST             = []                    # List to accumulate entities for Rasa's domain.yml
INTENT_LIST             = []                    # List to accumulate intents for Rasa's domain.yml
RULES_CONTENT           = ''                    # String to accumulate steps for Rasa's rules.yml

# Compile the regular expression pattern
usersays_pattern = re.compile(r'_usersays_.*\.json$')

#-----------------------------------------------------------------------------------------------
# Auxiliary Functions
#-----------------------------------------------------------------------------------------------
# This section includes utility functions for text processing, file manipulation, and configuration
# management. These functions are designed to support the main functionality of the script by
# providing reusable logic for common operations such as file content replacement, text normalization,
# and dynamic configuration based on user input or file data.

def set_language_setting():
    """
    Sets the global language setting based on the Dialogflow agent configuration.

    This function reads the `language` field from the `agent.json` file located within
    the Dialogflow input directory and updates the global `LANGUAGE_SETTING` variable.

    Raises:
        FileNotFoundError: If the 'agent.json' file does not exist within the Dialogflow input directory.
        json.JSONDecodeError: If there is an error parsing the 'agent.json' file.
    """
    global LANGUAGE_SETTING

    print('Setting language...')
    try:
        with open(os.path.join(DIALOGFLOW_INPUT_DIR, 'agent.json'), 'r', encoding='utf-8') as file:
            agent_data = json.load(file)
        LANGUAGE_SETTING = agent_data.get('language')

        if not LANGUAGE_SETTING:
            raise ValueError('Language setting is missing in the agent.json file.')
    except FileNotFoundError:
        print(f"Error: The 'agent.json' file was not found in {DIALOGFLOW_INPUT_DIR}.")
        raise
    except json.JSONDecodeError:
        print("Error: There was an issue parsing the 'agent.json' file.")
        raise
    
    print(f"Language setting set to '{LANGUAGE_SETTING}'.\n")


def is_usersays(filename: str) -> bool:
    """
    Determines whether the specified filename corresponds to a Dialogflow 'usersays' JSON file.

    Dialogflow exports 'usersays' data in files that follow a specific naming pattern, ending with
    '_usersays_{language}.json'. This function checks if the given filename matches this pattern,
    indicating it contains user expressions for an intent.

    Parameters:
    - filename (str): The filename to check.

    Returns:
    - bool: True if the filename matches the '_usersays_{language}.json' pattern, False otherwise.
    """
    global usersays_pattern

    # Use the compiled pattern for matching
    return usersays_pattern.search(filename) is not None


def extract_groups(filenames: list) -> list:
    """
    Identifies groups of intents based on the prefix of filenames.

    This function processes a list of Dialogflow intent filenames, grouping them by prefixes
    separated by a '-'. A group is considered valid if it contains at least two intents.
    Intents without a prefix are grouped into a default group if there are at least two such intents.

    Parameters:
    - filenames (list of str): The list of filenames to process.

    Returns:
    - list of str: A list of identified group prefixes, including the default group if applicable.
    """
    print('Extracting groups...')

    # Dictionary to count occurrences of each prefix
    prefix_count = {}

    no_prefix_count = 0

    for filename in filenames:
        # Skip non-usersays files
        if not is_usersays(filename):
            continue

        # Attempt to split each filename by the '-' character to identify the prefix
        parts = filename.split('-', 1)
        
        # If there is a prefix
        if len(parts) > 1:  
            prefix = parts[0]

            # Increment the count for the prefix, initializing if not already present
            prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
        else:
            no_prefix_count += 1
    
    # Filter out prefixes that appear at least twice
    groups = [prefix for prefix, count in prefix_count.items() if count >= 2]

    # Count items in prefix_count with a count less than 2
    no_prefix_count += sum(1 for _, count in prefix_count.items() if count < 2)

    # If there are at least 2 such items, add DEFAULT_GROUP to groups
    if no_prefix_count >= 2:
        groups.append(DEFAULT_GROUP)    

    # If there is only one group as intent, than we train each intent as independent
    # It cannot be trained with only one group
    if len(groups) == 1 and no_prefix_count == 0:
        groups = []

    if not groups:
        print('No groups found.\n')
    else:
        print(f'Extracted groups: {groups}.\n')

    return groups


def handle_empty_file(file_path: str):
    """
    Prompts the user to decide whether to delete an empty intent file and its corresponding 'usersays' file.

    Parameters:
    - file_path (str): The path to the intent file considered empty.
    """
    no_yes_list = ['n', 'N', 'no', 'No', 'NO', 'y', 'Y', 'yes', 'Yes', 'YES']
    no_list = ['n', 'N', 'no', 'No', 'NO']

    while True:
        inp = input('Delete empty file and its corresponding usersays? [y/n]: ')

        if inp in no_yes_list:
            if inp in no_list:
                print(f'Please delete or move {file_path} and its corresponding usersays manually. Then run the script again.\n')
                sys.exit()
            else:
                # Remove the intent file
                os.remove(file_path)

                # Construct and remove the corresponding 'usersays' file path
                usersays_file_path = file_path.replace('.json', f'_usersays_{LANGUAGE_SETTING}.json')

                if os.path.exists(usersays_file_path):
                    os.remove(usersays_file_path)

                print(f'Removed: {file_path} and its corresponding usersays file.')
            break

        print(f'Error. "{inp}" not an option.\n')


def ok_empty_files():
    """
    Identifies and optionally deletes Dialogflow intent files without responses
    and their corresponding 'usersays' files.

    This function iterates over all intent files in the Dialogflow intents directory,
    checks if an intent file lacks text responses, and prompts the user to delete both
    the intent file and its corresponding 'usersays' file if empty.
    """
    print('Checking for empty intent files...')

    empty_files_found = False

    for response_file in os.listdir(DIALOGFLOW_INTENTS_DIR):
        # Skip 'usersays' files
        if is_usersays(response_file):
            continue

        # Construct the full path for the file
        file_path = os.path.join(DIALOGFLOW_INTENTS_DIR, response_file)
        
        # Load the intent's data
        with open(file_path, 'r', encoding='utf-8') as file:
            response_data = json.load(file)

        # Check if the file has any non-empty text response
        if not any(message.get('speech') for response in response_data['responses'] 
                   for message in response.get('messages', []) if message.get('type') == '0'):
            print(f'FILE WITHOUT RESPONSE: {response_file}.')
            handle_empty_file(file_path)
            empty_files_found = True
    
    if not empty_files_found:
        print('No empty intent files found.\n')
    else:
        print('Finished processing empty intent files.\n')


def ignore_sys():
    """
    Prompts the user to decide whether to ignore Dialogflow's default system entities.

    This decision affects how entities are processed during the migration to Rasa,
    specifically whether system entities prefixed with 'sys.' should be included or excluded.

    The global variable `IGNORE_SYSTEM_ENTITIES` is set based on user input.
    """
    global IGNORE_SYSTEM_ENTITIES

    no_yes_list = ['n', 'N', 'no', 'No', 'NO', 'y', 'Y', 'yes', 'Yes', 'YES']
    no_list = ['n', 'N', 'no', 'No', 'NO']
    
    while True:
        inp = input("Do you want to ignore the default system files of Dialogflow (sys.)? [y/n]: ")
        print('\n')

        if inp in no_yes_list:         
            IGNORE_SYSTEM_ENTITIES = inp not in no_list
            break
        
        print(f'Error. "{inp}" not an option.\n')


def remove_substring(s: str, substr: str) -> str:
    """
    Removes a substring from the end of a string if it exists.

    Parameters:
    - s (str): The original string from which to remove the substring.
    - substr (str): The substring to remove.

    Returns:
    - str: The modified string with the substring removed if it was found at the end; otherwise, the original string.
    """
    if s.endswith(substr):
        return s[:-len(substr)]
    return s


def add_group_to_intent_name(intent_name: str) -> str:
    """
    Modifies the intent name by appending the longest matching group prefix from INTENT_LIST with a slash ('/') or
    prepends the DEFAULT_GROUP prefix if no match is found and DEFAULT_GROUP was added to INTENT_LIST.

    Parameters:
    - intent_name (str): The name of the intent to modify.

    Returns:
    - str: The modified intent name with the group prefix.
    """
    # Find the longest matching group prefix in INTENT_LIST
    matched_groups = [group for group in INTENT_LIST if intent_name.startswith(group)]

    if matched_groups:
        intent_group = max(matched_groups, key=len)
        # Replace the first occurrence of '-' with '/' after the longest matching group
        intent_name = intent_name.replace(intent_group + '-', intent_group + '/', 1)
    elif DEFAULT_GROUP in INTENT_LIST:
        # Prepend DEFAULT_GROUP only if it is explicitly part of INTENT_LIST
        intent_name = DEFAULT_GROUP + '/' + intent_name

    return intent_name


def format_text(text: str) -> str:
    """
    Standardizes quotation marks within a given text to double quotes and converts newline characters to spaces.

    Parameters:
    - text (str): The text to be normalized.

    Returns:
    - str: The normalized text with standardized quotation marks and spaces instead of newline characters.
    """
    # Standardize single quotes to double quotes
    text = text.replace("'", '\'')
    
    # Standardize left and right double quotation marks to plain double quotes
    text = text.replace('“', '"').replace('”', '"')
    
    # Convert newline characters to spaces
    text = text.replace('\n', ' ')
    
    # Consider adding a step to remove consecutive spaces resulting from newline replacements,
    # if applicable to your use case:
    # text = ' '.join(text.split())
    
    return text


def replace_in_file(filename: str, old_text: str, new_text: str):
    """
    Replaces all occurrences of a specified text within a file with new text.

    This function reads the contents of a file, performs a text replacement operation,
    and writes the modified content back to the same file.

    Parameters:
    - filename (str): The path to the file to modify.
    - old_text (str): The text within the file that needs to be replaced.
    - new_text (str): The text to replace occurrences of `old_text`.
    """
    # Attempt to read in the file
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            filedata = file.read()
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        return

    # Perform the replacement
    filedata = filedata.replace(old_text, new_text)

    # Attempt to write the modified content back to the file
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(filedata)
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")


def rename_intent(intent_name: str) -> str:
    """
    Rename the intent name by removing its group prefix (if any) and update the 'nlu.yml' file to reflect
    the new intent name. The updated intent name is also added to the global INTENT_LIST.

    Parameters:
    - intent_name (str): The original name of the intent.

    Returns:
    - str: The updated intent_name, without the group prefix.
    """
    global INTENT_LIST

    old_name = intent_name
    # Extract the intent name after the '/' if it exists; otherwise, retain the original
    intent_name = intent_name.split('/', 1)[-1]

    # Check if the new intent name is already in the list to avoid duplicates
    if not intent_name.startswith(tuple(INTENT_LIST)):
        INTENT_LIST.append(intent_name)
    
    # Define the path to the 'nlu.yml' file
    nlu_file_path = os.path.join(RASA_DATA_DIR, 'nlu.yml')

    replace_in_file(nlu_file_path, old_name, intent_name)

    return intent_name


def rename_response(response_name: str, intent_name: str, responses_content: str) -> tuple[str, str]:
    """
    Renames the response_name based on the given intent_name and replaces the old name with the new one
    in the provided responses_content string.

    Parameters:
    - intent_name (str): The original name of the intent.
    - response_name (str): The original name of the response to be renamed.
    - responses_content (str): The string containing all responses from the domain file.

    Returns:
    - Tuple[str, str]: The updated response_name and responses_content.
    """
    old_resp = response_name
    # Construct the new response name with a prefix
    response_name = f'utter_{intent_name}'

    # Replace the response name in domain
    responses_content = responses_content.replace(old_resp, response_name)

    return response_name, responses_content


def create_new_rule(rules: str, response_name: str, intent_name: str) -> str:
    """
    Appends a new rule to the existing Rasa rules. Each rule is formatted to indicate a direct
    mapping between an intent and its associated action (response).

    Parameters:
    - rules (str): The current content of Rasa rules.
    - response_name (str): The name of the response.
    - intent_name (str): The name of the intent associated with the response.

    Returns:
    - str: The updated rules with the new rule.
    """
    # Format for a new rule in Rasa involves specifying the rule name, steps including the intent and action
    new_rule = (
        f"\n- rule: Respond to {intent_name}\n"
        f"  steps:\n"
        f"  - intent: {intent_name}\n"
        f"  - action: {response_name}\n"
    )

    # Append the new rule to the existing rules
    rules += new_rule

    return rules


def process_buttons(payload, responses_content, after_text, steps_iter, intent_name, response_name):
    """
    Processes button payloads from Dialogflow's response messages for Rasa format, updating the
    responses content, and potentially the rules content to include button actions as steps.
  
    Parameters:
    - payload (dict): The payload dictionary from a Dialogflow message.
    - responses_content (str): The current string of formatted responses content.
    - after_text (bool): A flag indicating if there's preceding text content.
    - steps_iter (int): An iterator for numbering steps in RULES_CONTENT.
    - intent_name (str): The name of the current intent being processed.
    - response_name (str): The formatted name for the Rasa response.
    
    Returns:
    - Tuple[str, int, bool, str, str]: Updated responses_content, steps_iter, after_text, intent_name and response_name.
    """
    global RULES_CONTENT

    if after_text:
        # Remove last '\n from domain
        responses_content = responses_content[:-2]
    else:
        # Create new text response to add URL:
        if steps_iter == 0:
            # Rename intent and replace it in nlu.yml
            intent_name = rename_intent(intent_name)
            # Rename response and replace it in responses_content
            response_name, responses_content = rename_response(response_name, intent_name, responses_content)
            # Create new response (rule and 1st step(previous text))
            RULES_CONTENT = create_new_rule(RULES_CONTENT, response_name, intent_name)
                            
        # if the rule already exists, add the next steps
        steps_iter += 1
        RULES_CONTENT += f'  - action: {response_name}_{steps_iter}\n'

        # Start text
        responses_content += f'  {response_name}_{steps_iter}:\n  - text: \''

    # Write url in domain text
    for url in payload.get('buttons'):
        # Format “, ', \n 
        url_text = format_text(url.get('text'))
        # Get url
        url_value = url.get('value')

        # Add to domain
        responses_content += f'<br><a href="{url_value}">{url_text}</a>'

    # Add removed '\n
    responses_content += "'\n"
    after_text = True

    return responses_content, steps_iter, after_text, intent_name, response_name


def process_images(payload: dict, responses_content: str, after_text: bool) -> tuple[str, bool]:
    """
    Processes image payloads from Dialogflow's response messages and formats them for Rasa,
    appending image URLs to the responses content.

    Parameters:
    - payload: The payload dictionary from a Dialogflow message.
    - responses_content: The current string of formatted responses content.
    - after_text: A flag indicating if there's preceding text content.
    
    Returns:
    - Tuple[str, bool]: Updated responses_content and after_text.
    """
    for image in payload.get('images'):
        image_url = image.get('imageUrl')
        responses_content += f'    image: "{image_url}"\n'

    after_text = False
    return responses_content, after_text


def get_responses():
    """
    Extracts responses from Dialogflow intent files and formats them for Rasa's domain file. This function
    iterates over all Dialogflow intent files, excluding the ones designated for user expressions (userSays),
    to compile a comprehensive list of responses and format them according to Rasa's requirements.
    It handles different types of response messages, including text, buttons, and images, and organizes
    them under corresponding response names.

    Returns:
    - str: A formatted string containing all responses for inclusion in Rasa's domain file.
    """
    responses_content = 'responses:\n'

    for response_file in os.listdir(DIALOGFLOW_INTENTS_DIR):
        if not is_usersays(response_file):
            with open(os.path.join(DIALOGFLOW_INTENTS_DIR, response_file), 'r', encoding='utf-8') as file:
                response_data = json.load(file)
            
            intent_name = remove_substring(response_file, '.json')
            intent_name = add_group_to_intent_name(intent_name)
            response_name = f'utter_{intent_name}'

            responses_content += f'  {response_name}:\n'
            response = response_data['responses'][0]

            # Initialize variable to manage rule steps
            after_text = False
            steps_iter = 0

            for message in response['messages']:
                # If message language is incorrect
                # than skip this message
                if message.get('lang') != LANGUAGE_SETTING:
                    continue
                if message.get('type') == '0' and message.get('speech'):
                    for example in message['speech']:
                        example = format_text(example)
                        responses_content += f"  - text: '{example}'\n"
                        after_text = True

                elif message.get('type') == '4':
                    payload = message.get('payload', {})
                    if 'buttons' in payload:
                        responses_content, steps_iter, after_text, intent_name, response_name = process_buttons(payload, responses_content, after_text, steps_iter, intent_name, response_name)

                    if 'images' in payload:
                        responses_content, after_text = process_images(payload, responses_content, after_text)

    return responses_content


#-----------------------------------------------------------------------------------------------
# Main Functions
#-----------------------------------------------------------------------------------------------
def generate_synonyms():
    """
    Generates a synonyms.yml file for Rasa NLU from Dialogflow entity files. This function iterates over
    Dialogflow entity files, extracts entities and their synonyms, and writes them to a synonyms.yml file
    in the Rasa NLU format. It updates the global ENTITY_LIST with the names of the processed entities.
    
    Global variables used:
    - RASA_DATA_DIR: Directory to save the synonyms.yml file.
    - DIALOGFLOW_ENTITIES_DIR: Directory containing Dialogflow entity files.
    - ENTITY_LIST: List of entity names to be updated with the names of processed entities.
    - IGNORE_SYSTEM_ENTITIES: Flag to ignore system entities from Dialogflow.
    - LANGUAGE_SETTING: Specifies the language of the entity files to be processed.
    """
    global ENTITY_LIST

    synonyms_file_path = os.path.join(RASA_DATA_DIR, 'synonyms.yml')

    try:
        # Initiate synonyms file
        with open(synonyms_file_path, 'w', encoding='utf-8') as synonyms_file:
            synonyms_file.write('nlu:\n')

            print(f'Generating {synonyms_file.name}...')

            for entity_file in os.listdir(DIALOGFLOW_ENTITIES_DIR):
                # Skip system entities if IGNORE_SYSTEM_ENTITIES is True
                if IGNORE_SYSTEM_ENTITIES and entity_file.startswith('sys.'):
                    continue
                
                # File with synonyms
                if entity_file.endswith(f'_entries_{LANGUAGE_SETTING}.json'):
                    entity_file_path = os.path.join(DIALOGFLOW_ENTITIES_DIR, entity_file)
                    
                    # Get entity data
                    with open(entity_file_path, 'r', encoding='utf-8') as file:
                        entity_data = json.load(file)

                    # Get entity name
                    entity_name = entity_data[0]['value']
                    ENTITY_LIST.append(entity_name)

                    # Write synonyms to file
                    synonyms_file.write(f'- synonym: {entity_name}\n  examples: |\n')

                    for value in entity_data:
                        # Store entitie's synonyms 
                        for synonym in value['synonyms']:
                            synonyms_file.write(f'    - {synonym}\n')
                    synonyms_file.write('\n')

            print('Done.\n')  
    except Exception as e:
        print(f'An error occurred while generating synonyms: {e}')      


def generate_nlu():
    """
    Generates the nlu.yml file for Rasa by converting Dialogflow intent files. This function reads Dialogflow
    intent files, formats them according to Rasa NLU training data format, and updates the nlu.yml file in
    the Rasa project directory.
    
    Global variables used:
    - RASA_DATA_DIR: Directory to save the nlu.yml file.
    - DIALOGFLOW_INTENTS_DIR: Directory containing Dialogflow intent files.
    - INTENT_LIST: List of intent names to be updated with the names of processed intents.
    - DEFAULT_GROUP: Default group to categorize intents.
    """
    global INTENT_LIST 
    
    print(f'Generating {RASA_DATA_DIR}/nlu.yml...')

    # Initialize variable to accumulate NLU data
    new_nlu_data = 'nlu:\n'

    try:

        for intent_file in os.listdir(DIALOGFLOW_INTENTS_DIR):
            # File with intent's examples
            if intent_file.endswith(f'_usersays_{LANGUAGE_SETTING}.json'):
                # Get intent's data
                with open(os.path.join(DIALOGFLOW_INTENTS_DIR, intent_file), 'r', encoding='utf-8') as file:
                    intent_data = json.load(file)

                # Create intent name
                intent_name = remove_substring(intent_file, f'_usersays_{LANGUAGE_SETTING}.json')
                intent_name = add_group_to_intent_name(intent_name) 

                # Ensure unique intent names are collected
                if not intent_name.startswith(tuple(INTENT_LIST)):
                    INTENT_LIST.append(intent_name)
                    
                # Start intent block
                new_nlu_data += f'- intent: {intent_name}\n  examples: |\n'

                # Write examples
                for example in intent_data:
                    text_data = ''.join([text_part['text'] for text_part in example['data']])    
                    new_nlu_data += f'    - {text_data}\n'

                new_nlu_data += '\n'

        # Writing or updating the NLU file with new data
        with open(os.path.join(RASA_DATA_DIR, 'nlu.yml'), 'r') as nlu_file:
            nlu_data = nlu_file.read()

        # Replace old data
        start_index = nlu_data.find('nlu:\n')

        # Replace the part of the original string
        nlu_data = nlu_data[:start_index] + new_nlu_data
        
        # Write the file out again
        with open(os.path.join(RASA_DATA_DIR, 'nlu.yml'), 'w') as nlu_file:
            nlu_file.write(nlu_data)

        print('Done.\n')    
    except Exception as e:
        print(f'An error occurred while generating NLU data: {e}')


def generate_domain():
    """
    Generates or updates the domain.yml file for a Rasa project. This function updates the intents,
    entities, and responses sections of the domain.yml file based on the global INTENT_LIST and ENTITY_LIST,
    and additional responses generated by get_responses().

    Global variables used:
    - RASA_OUTPUT_DIR: Directory where the domain.yml file is located.
    - INTENT_LIST: List of intent names to be included in the domain.yml file.
    - ENTITY_LIST: List of entity names to be included in the domain.yml file.
    """
    domain_file_path = os.path.join(RASA_OUTPUT_DIR, 'domain.yml')
    print(f'Generating {domain_file_path}...')

    try:
        # Initialize domain contents
        responses_content = get_responses()

        intents_content = 'intents:\n' + '\n'.join([f'- {intent}' for intent in INTENT_LIST]) + '\n\n'
        entities_content = 'entities:\n' + '\n'.join([f'- {entity}' for entity in ENTITY_LIST]) + '\n\n'

        # Attempt to read the existing domain file
        try:
            with open(domain_file_path, 'r') as file:
                filedata = file.read()
        except FileNotFoundError:
            print(f"{domain_file_path} not found. A new file will be created.")
            filedata = 'version: "3.1"\n'

        # Determine where to insert new content; if 'intents:\n' is not found, append at the end
        start_index = filedata.find('intents:\n')
        if start_index == -1:
            filedata += intents_content + entities_content + responses_content
        else:
            filedata = filedata[:start_index] + intents_content + entities_content + responses_content

        # Write the updated content back to the domain file
        with open(domain_file_path, 'w') as file:
            file.write(filedata)

        print('Done.\n')
    except Exception as e:
        print(f'An error occurred while generating the domain file: {e}')


def generate_rules():
    """
    Generates or updates the rules.yml file for a Rasa project, incorporating predefined rules
    content and automatically generating new rules for intents not already covered.
    """
    rules_file_path = os.path.join(RASA_DATA_DIR, 'rules.yml')
    print(f'Generating {rules_file_path}...')
    
    # Start with predefined rules content, or initialize if empty
    new_rules = f'rules:\n{RULES_CONTENT}'

    for intent in INTENT_LIST:
        # Skip intent if a rule already exists
        if intent in RULES_CONTENT:
            continue

        # Append default rule for intent
        new_rules += (
            f'\n- rule: Respond to {intent}\n'
            '  steps:\n'
            f'  - intent: {intent}\n'
            f'  - action: utter_{intent}\n'
        )

    try:
        # Try to read the existing rules file if it exists to preserve any manual edits
        try:
            with open(rules_file_path, 'r') as rules_file:
                rules_data = rules_file.read()

            # Find the start of the rules block to append new rules
            start_index = rules_data.find('rules:\n')
            if start_index != -1:
                # If existing rules are found, append after the existing content
                rules_data = rules_data[:start_index] + new_rules
            else:
                # If 'rules:\n' is not found, start fresh
                rules_data = f'version: "3.1"\n{new_rules}'
        except FileNotFoundError:
            # If the file doesn't exist, start with new_rules
            rules_data = f'version: "3.1"\n{new_rules}'
        
        # Write the updated or new rules content back to the file
        with open(rules_file_path, 'w') as rules_file:
            rules_file.write(rules_data)         

        print('Done.')
    except Exception as e:
        print(f'An error occurred while generating the rules file: {e}')


def initialize_migration_setup():
    """
    Prepares the environment for the migration process by setting the language setting,
    extracting intent groups, checking for and handling empty intent files, and determining
    whether to ignore system entities.

    Global Variables Modified:
    - INTENT_LIST: Populated with the list of intent groups extracted from the Dialogflow intent directory.

    Side Effects:
    - May terminate the program if an empty file is found and not deleted upon prompt.
    - Sets global configuration flags based on user input and file analysis.
    """
    global INTENT_LIST

    set_language_setting()
    INTENT_LIST = extract_groups(os.listdir(DIALOGFLOW_INTENTS_DIR))
    ok_empty_files()
    ignore_sys()


def main():    
    """
    Orchestrates the migration setup and the generation of various Rasa configuration files
    including synonyms, NLU data, domain information, and rules based on the provided
    Dialogflow data.
    """
    initialize_migration_setup()
    generate_synonyms()
    generate_nlu()
    generate_domain()
    generate_rules()
    

if __name__ == "__main__":
    # Call main
    main()