import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import Counter
import re
import json
import copy
import pandas as pd


##################### SPLITTING FUNCTIONS #####################

def split_ands_phases(sentence_data):
    """Splits any occurence of the symbol && used to denote two different phases."""
    sentence_data_res = []
    for s in sentence_data:
        s['phase'] = s['phase'].strip()

        if "&&" in s['phase']:
            phases_in_s = s['phase'].split("&&")
            for p in phases_in_s:
                s_to_duplicate = s.copy()
                s_to_duplicate['phase'] = p.strip()
                sentence_data_res.append(s_to_duplicate)
        else:    
            sentence_data_res.append(s) 
            
    return sentence_data_res

def split_commas_phases(sentence_data):
    """
    Splits any occurence of commas used to denote two different phases.
    Avoids commas between parantheses such as Al3(Sc,Zr).
    """
    sentence_data_res = []
    commas_split_patt = re.compile(r',\s*(?![^()]*(\)|]))')
    
    for s in sentence_data:
        s['phase'] = s['phase'].strip()

        if "," in s['phase']:
            phases_in_s = [i for i in re.split(commas_split_patt, s['phase']) if i is not None]
            for p in phases_in_s:
                s_to_duplicate = s.copy()
                s_to_duplicate['phase'] = p.strip()
                sentence_data_res.append(s_to_duplicate)
        else:    
            sentence_data_res.append(s) 
            
    return sentence_data_res

def split_ands_props(sentence_data):
    """Splits any occurence of the symbol && used to denote two different properties."""
    sentence_data_res = []
    for s in sentence_data:
        s['property'] = s['property'].strip()

        if "&&" in s['property']:
            props_in_s = s['property'].split("&&")
            for p in props_in_s:
                s_to_duplicate = s.copy()
                s_to_duplicate['property'] = p.strip()
                sentence_data_res.append(s_to_duplicate)
        else:    
            sentence_data_res.append(s) 
            
    return sentence_data_res

def split_commas_props(sentence_data):
    """Splits any occurence of commas used to denote two different properties."""
    sentence_data_res = []
    commas_split_patt = re.compile(r',\s*(?![^()]*(\)|]))')
    
    for s in sentence_data:
        s['property'] = s['property'].strip()

        if "," in s['property']:
            props_in_s = [i for i in re.split(commas_split_patt, s['property']) if i is not None]
            for p in props_in_s:
                s_to_duplicate = s.copy()
                s_to_duplicate['property'] = p.strip()
                sentence_data_res.append(s_to_duplicate)
        else:    
            sentence_data_res.append(s) 
            
    return sentence_data_res

def split_entities(sentence_data):
    """Evaluate all split functions at the same time"""
    sentence_data_res = copy.deepcopy(sentence_data)
    
    sentence_data_res = split_ands_phases(sentence_data_res)
    sentence_data_res = split_commas_phases(sentence_data_res)
    sentence_data_res = split_ands_props(sentence_data_res)
    sentence_data_res = split_commas_props(sentence_data_res)

    return sentence_data_res


##################### ENTITY AGGREGATION CLEANING FUNCTIONS #####################

def remove_change_words_phase(sentence_data, hard_stop_words_phase, soft_stop_words_phase, elem_names_dict):
    """
    Acts on words in phases, specifically to change or remove them. Steps performed:
    1. Replaces greek and quotation marks
    2. Removes whole mention if any word is in hard_stop_words
    3. Removes any words if they appear in soft_stop_words
    4. Transforms element names into symbols
    5. Any common mention of Si phase is transformed as such
    """
    sentence_data_res = copy.deepcopy(sentence_data)
    mask_to_keep = np.ones(len(sentence_data_res), dtype=bool)
        
    for i_s, s in enumerate(sentence_data_res):
        #replace some common symbols
        s['phase'] = s['phase'].replace("″", "''")
        s['phase'] = s['phase'].replace("′", "'")
        s['phase'] = s['phase'].replace('"', "''")
        
        s['phase'] = re.sub(r"\bbeta\b", "β", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bdelta\b", "δ", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\beta\b", "η", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\btheta\b", "θ", s['phase'], flags=re.IGNORECASE)
        
        s['phase'] = re.sub(r"\bb'\B", "β'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bb '\B", "β'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bb''\B", "β''", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bb ''\B", "β''", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bb\b", "β", s['phase'], flags=re.IGNORECASE)
        
        s['phase'] = re.sub(r"\bd'\B", "δ'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bd '\B", "δ'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bd''\B", "δ''", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bd ''\B", "δ''", s['phase'], flags=re.IGNORECASE)
        
        s['phase'] = re.sub(r"\be'\B", "η'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\be '\B", "η'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\be''\B", "η''", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\be ''\B", "η''", s['phase'], flags=re.IGNORECASE)
        
        s['phase'] = re.sub(r"\bth'\B", "θ'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bth '\B", "θ'", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bth''\B", "θ''", s['phase'], flags=re.IGNORECASE)
        s['phase'] = re.sub(r"\bth ''\B", "θ''", s['phase'], flags=re.IGNORECASE)
        
        #split into each word
        phase_words = re.split(' |-', s['phase'])
        
        #if any words are in our remove list then we do that
        if any([pw.lower() in hard_stop_words_phase for pw in phase_words]):
            mask_to_keep[i_s] = False
            continue
        
        for pw in phase_words.copy():
            #if the word is in our list to remove, we do that
            if pw in soft_stop_words_phase:
                phase_words.remove(pw)
           
        for pw_i, pw in enumerate(phase_words):
            #if there is a element name, replace with symbol
            if pw.lower() in elem_names_dict:
                phase_words[pw_i] = elem_names_dict[pw.lower()]
        
        s['phase'] = " ".join(phase_words).strip()
        
        if "si" in phase_words and ("eutectic" in phase_words or "phase" in phase_words or "primary" in phase_words):
            s['phase'] = "si [to_keep]"

    return np.array(sentence_data_res)[mask_to_keep].tolist()


def remove_change_words_prop(sentence_data, hard_stop_words_prop, soft_stop_words_prop):
    """
    Acts on words in properties, specifically to change or remove them.
    """
    sentence_data_res = copy.deepcopy(sentence_data)
    mask_to_keep = np.ones(len(sentence_data_res), dtype=bool)
        
    for i_s, s in enumerate(sentence_data_res):
        #replace some common symbols
        s['property'] = s['property'].replace('strengthening', "hardening")
        s['property'] = s['property'].replace('-', " ")
        
        #split into each word
        prop_words = s['property'].split(" ")
        
        #if any words are in our remove list then we do that
        if any([pw.lower() in hard_stop_words_prop for pw in prop_words]):
            mask_to_keep[i_s] = False
            continue
        
        for pw_i, pw in enumerate(prop_words.copy()):
            
            #if the word is in our list to remove, we do that
            if pw in soft_stop_words_prop:
                prop_words.remove(pw)
                continue
        
        s['property'] = " ".join(prop_words).strip()

    return np.array(sentence_data_res)[mask_to_keep].tolist()


def merge_chem_names(sentence_data, chem_names):
    """Finds, anywhere in the string, for a chemical name and changes the string to match it."""
    sentence_data_res = copy.deepcopy(sentence_data)

    for s in sentence_data_res:
        for cnames in chem_names:
            for cn in cnames:
                patt = re.compile(rf"\b{cn}\b")
                if re.search(patt, s['phase']) is not None:
                    if "''" in s['phase']:
                        s['phase'] = f"doublemeta-{cnames[0]}"
                    elif "'" in s['phase']:
                        s['phase'] = f"meta-{cnames[0]}"
                    else:
                        s['phase'] = cnames[0]
            
    return sentence_data_res


def remove_change_full(sentence_data, elem_names_dict, phase_to_remove, phase_map, prop_to_remove, prop_map):
    """
    Acts on the full phase or property name. Steps performed:
    1. Removal of all simple element names
    2. Removal of properties and phases in the to_remove lists
    3. Removes any relationship that is neither good or bad
    4. Map phases and properties according to dictionnary
    """
    sentence_data_res = copy.deepcopy(sentence_data)
    mask_to_keep = np.ones(len(sentence_data_res), dtype=bool)
        
    for i_s, s in enumerate(sentence_data_res):
        s['phase'] = s['phase'].strip()
        s['property'] = s['property'].strip()
        
        #remove any phase name that is just an element symbol
        if s['phase'].lower() in [v.lower() for v in elem_names_dict.values()]:
            mask_to_keep[i_s] = False
            continue

        #removes any phases or properties that are in the to_remove lists
        if s['phase'].lower() in phase_to_remove or s['property'].lower() in prop_to_remove or len(s['phase'])==0 or len(s['property'])==0:
            mask_to_keep[i_s] = False
            continue

        #removes any relationship that is neither good or bad
        s['relationship'] = s['relationship'].lower()
        if s['relationship'] not in ['bad', 'good']:
            mask_to_keep[i_s] = False
            continue

        #map phases in the dictionnary
        if s['phase'].lower() in phase_map:
            s['phase'] = phase_map[s['phase'].lower()]

        #map properties in the dictionnary
        if s['property'].lower() in prop_map:
            s['property'] = prop_map[s['property'].lower()]

    return np.array(sentence_data_res)[mask_to_keep].tolist()


def check_beta_paragraphs(sentence_data, possible_beta):
    sentence_data_res = copy.deepcopy(sentence_data)
    chem_patt = re.compile(r"\b(?:[A-Z][a-z]?\d*)+\b")
    
    for s in sentence_data_res:
        if s['phase']=="β":
            phases_found = re.findall(chem_patt, s['paragraph'])
            in_para = [p in phases_found for p in possible_beta]
            if sum(in_para)==1:
                s['phase'] = np.array(list(possible_beta.values()))[in_para][0]
            else:
                s['phase'] = "β-unknown"
            
    return sentence_data_res

def post_processing(sentence_data, post_process_map):
    """Give back the old names of certain phases after processing"""
    sentence_data_res = copy.deepcopy(sentence_data)
    
    for s in sentence_data_res:
        if s['phase'] in post_process_map.keys():
            s['phase'] = post_process_map[s['phase']]
            
    return sentence_data_res


##################### PLOTTING FUNCTIONS #####################

def plot_sentiment(prop, data, n=5, save=False):
    """
    Plots, the the top n phase, the sentiments of seach phase associated with the property prop.
    
    Args:
        prop: (string) Property to investigate.
        data: (list of dict) The data to use when plotting.
        n: (int, default: 5) Number of phases to look at.
        save: (string, default: False) If not False, this is the string of where to save the plot.
    """
    
    # Get the number of mentions for each phase with the prop
    counted = Counter([s['phase'] for s in data if s['property']==prop])
    phases = list(dict(sorted(counted.items(), key=lambda item: item[1], reverse=True)).keys())[:n]

    # Get the relationship between each phase and prop
    sentiment = {}
    all_mentions = [s['relationship'] for s in data if s['property']==prop]
    for p in phases:
        sentiment[p] = [s['relationship'] for s in data if s['phase']==p if s['property']==prop]

    # Count the number of positive and negative mentions
    goods = [(np.array(sentiment[p])=='positive').sum()/len(sentiment[p])*100 for p in sentiment.keys()]
    good_counts = [(np.array(sentiment[p])=='positive').sum() for p in sentiment.keys()]
    bads = [(np.array(sentiment[p])=='negative').sum()/len(sentiment[p])*100 for p in sentiment.keys()]
    bad_counts = [(np.array(sentiment[p])=='negative').sum() for p in sentiment.keys()]

    # Create plot
    fig, ax = plt.subplots(figsize=(12/5*n, 7))

    cmap = matplotlib.colormaps['RdYlGn']

    ax.barh(phases, goods, align='center', color=cmap(255))
    ax.barh(phases, bads, left=goods, align='center', color=cmap(0))

    ax.axvline((np.array(all_mentions)=='positive').sum()/len(all_mentions)*100, color='black', linestyle='--', linewidth=3)

    # Add number of mentions on either side of the plot
    for i, p in enumerate(sentiment.keys()):
        ax.text(2, i, good_counts[i], va='center', ha='left', fontsize=30, color='white', weight="bold", zorder=50)
        ax.text(98, i, bad_counts[i], va='center', ha='right', fontsize=30, color='white', weight="bold", zorder=50)

    ax.tick_params(labelsize=30)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel('Positive mentions (%)', fontsize=30)
    ax.set_title(f"{prop.capitalize()}", fontsize=30)

    fig.tight_layout()
    
    # If we want to save the plot, we do it here
    if save: fig.savefig(save, dpi=150)