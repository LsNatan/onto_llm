import argparse
import os
from collections import defaultdict
import csv
import requests
import subprocess
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openai
import tqdm

from rdfox_app import RDFOXApp



with open('openai_key.txt') as f:
    openai.api_key = f.read()

def create_rules_from_prompt(ontology_file, rules_file):
    user_text = input("Enter you prompt here:")
    free_text_request = f"""write sparql selection query which selects the following objects:
    {user_text}.
    write the query as if the rules are not loaded to RDFOX
     The name of the point of interest should  be used as literal string"""
    prefix = 'Answer as an expert computer scientist.:\n'
    suffix = f"""
Using only the given ontology and set of rules, {free_text_request}
Your response should only include the code, without any explanations or annotations.
    """
    with open(ontology_file) as f:
        ontology = f.read()

    with open(rules_file) as f:
        rules = f.read()

    prompt = prefix + ontology +'\n' + rules + suffix

    message = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=0.2,
        max_tokens=100,
        top_p=0.2,
        frequency_penalty=0.0,
        presence_penalty=0.2,
    )
    selection_query = response["choices"][0]["message"]["content"]
    # selection_query = selection_query.split("\n")
    print(f'{"*"*60}\n LLM Prompt: \n {"*"*60} \n {prompt}')
    print(f'{"*"*60}\n LLM response: \n {"*"*60} \n {selection_query}')
    # print(f'rules list {rules_list}')
    # rules_path = os.path.join(Path(ontology_file).parent, 'rules.dlog')
    # with open(rules_path, 'w') as f:
    #     f.write(parsed_response)

    return selection_query

def run_nerratives(args, rdfox_server, narrative_selection_query=None):






    def complex_rels_converter(data):
        if '{' in data:
            d = data.split('\n')
            for i in d:
                if '{' in i:
                    i = complex_rel_converter(i)
            converted_data = ''
            for i in d:
                converted_data = converted_data + i + '\n\n'
        else:
            converted_data = data
        return converted_data

    def complex_rel_converter(complex_rel):  # N3 syntax e.g. p:Bill cog:thinks {p:John f:loves p:Mary} .
        b = complex_rel.split('{')
        b1 = b[1].split('}')
        statement = b1[0].split(' ')
        b2 = b[0].split(' ')
        if len(b2) > 1:
            data = b[0] + ' statement; statement rdf:type rdf:statement; statement rdf:object ' + b2[
                0] + ';statement rdf:predicate ' + b2[1] + 'statement rdf:subject ' + b2[0] + ';'
        else:
            b2 = b1[1].split(' ')
            data = 'statement ' + b1[1] + ' ;statement rdf:type rdf:statement; statement rdf:object ' + b2[
                0] + ';statement rdf:predicate ' + b2[1] + 'statement rdf:subject ' + b2[0] + ';'
        return data

    def insert_data(data):
        data = complex_rels_converter(data)
        st = ''
        for i in prefs:
            st = st + "\n@" + i + "."
        new_tar = st + '\n\n' + data + '\n \n'
        response = requests.post(rdfox_server1, data=new_tar)  # insert new target
        return response

    def delete_data(triples):
        st = ''
        for i in prefs:
            st = st + i + ". "
        datalog_text = st + " " \
                            ' ' + triples
        response = requests.patch(rdfox_server1, params={"operation": "delete-content"}, data=datalog_text)
        return response

    def query(query):
        st = ''
        for i in prefs:
            st = st + i
        sparql_text = st + \
                      query
        spl = query.split(' ')
        if spl[0][1:] == "ELECT":
            response = requests.get(rdfox_server2, params={"query": sparql_text})
        else:
            response = requests.get(rdfox_server2, params={"update": sparql_text})
        return response

    def rule(rule):
        st = ''
        for i in prefs:
            st = st + "\n@" + i + "."
        datalog_rule1 = st + \
                        rule
        response = requests.post(rdfox_server1, data=datalog_rule1)
        return response

    def extract_coor(d):
        dlist = d.text.split('\n')
        alon = []
        alat = []
        for i in dlist[1:-1]:
            pair = i.split('\t')
            alon.append(float(pair[1]))
            alat.append(float(pair[2]))
        return alon, alat

    def extract_data1(d, is_dat=0):
        dlist = d.text.split('\n')
        pairlen = len(dlist[0].split('\t'))
        list_of_pairs = []
        if is_dat == 1:
            dat = []
            for j in range(pairlen):
                dat.append([])
        for i in dlist[1:-1]:
            pair = i.split('\t')
            for i in list_of_IRI:
                pair[1] = pair[1].replace(i, '')
            pair[1] = pair[1].replace('>', '')
            for i in list_of_IRI:
                pair[2] = pair[2].replace(i, '')
            pair[2] = pair[2].replace('>', '')
            list_of_pairs.append(pair)
            if is_dat == 1:
                for ii in range(pairlen):
                    dat[ii].append(pair[ii])
        if is_dat == 1:
            return dat, list_of_pairs
        else:
            return list_of_pairs

    def give_prob(init_prob, list_of_pair, list_of_pair6, ruless):
        prolog_list1 = ruless
        beg = time.time()
        for ii, i in enumerate(list_of_pair):
            if 'type' in i[1]:
                prolog_list1 = prolog_list1 + str(init_prob[ii]) + '::' + i[2] + '(' + i[0] + ').\n'
            else:
                prolog_list1 = prolog_list1 + str(init_prob[ii]) + '::' + i[1] + '(' + i[0] + ',' + i[2] + ').\n'
        prolog_list2 = prolog_list1 + ''
        for jj, j in enumerate(list_of_pair6):
            if 'type' in j[1]:
                prolog_list2 = prolog_list2 + 'query(' + j[2] + '(' + j[0] + ')).\n'
            else:
                prolog_list2 = prolog_list2 + 'query(' + j[1] + '(' + j[0] + ',' + j[2] + ')).\n'
        p = PrologString(prolog_list2)
        ff = list(get_evaluatable().create_from(p).evaluate().values())[-len(list_of_pair6):]
        return np.array(ff)


    def get_info_from_track_numbers(targets_near_poi, targets_lat_lon):
        for target in targets_near_poi:
            query_string = f"""SELECT DISTINCT  ?lat ?lon
                                WHERE {{
                                  ?d :detection_of ?t ;
                                     :lat ?lat ;
                                     :lon ?lon .
                                  FILTER(?t={target})
                                }}"""
            res = query(query_string)
            dlist = res.text.split('\n')
            for i in dlist[1:-1]:
                lat, lon = i.split('\t')
                targets_lat_lon[target].append((lat, lon))
        return targets_lat_lon

    def save_targets_lat_lon_to_visualizer_format(targets_lat_lon, poi_data):
        poi_name, poi_lat, poi_lon = poi_data
        header = 'r/d route order name  long  lat color'.split()
        order = 0
        with open('for_visualization.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            for i, (target, detections) in enumerate(targets_lat_lon.items()):
                first_det_of_track = False
                color = str(np.random.random(3).tolist()).replace(',', "")
                for j, detection in enumerate(detections):
                    lat, long = detection
                    row = ['r', i, order, i if first_det_of_track else "", long, lat, color]
                    # first_det_of_track = False
                    # write the data
                    writer.writerow(row)
                    order+=1
            # Write POI
            color = '[0 1 0]'
            row = ['d', i+1, order, poi_name.strip('"'), poi_lon, poi_lat, color]
            writer.writerow(row)

    def extract_poi_name(narrative_selection_query):
        import re

        # Regular expression pattern
        pattern = r':poi_name\s+"([^"]+)"'

        # Input text to search for the pattern

        # Find all matches of the pattern in the text
        matches = re.findall(pattern, narrative_selection_query)

        # Print the matches
        for match in matches:
            return match

    # options = TypeDBOptions.core()
    # options.infer = True
    # insert at 1, 0 is the script path (or '' in REPL) # sys.path.insert(1, '/home/adiel/Desktop/grakn-core-all-linux-1.8.4/grakn-core-all-linux-1.8.4/')
    keyspace = "narratives"
    num_of_poi = 50
    num_of_chunks = 3
    chunk_size = 2000  # at most 2000
    to_ins = 1
    is_prob = 0
    near_par = 0.001
    operated_narratives = [6]
    # operated_narratives = [0, 1, 2, 3, 4, 5]
    # rdfox_server = "http://localhost:12110"
    response = requests.post(
        rdfox_server + "/datastores/" + keyspace, params={'type': 'par-complex-nn'})
    rdfox_server1 = rdfox_server + "/datastores/" + keyspace + "/content"
    rdfox_server2 = rdfox_server + "/datastores/" + keyspace + "/sparql"
    # list_of_IRI = ['https://www.semanticweb/adiel/ontologies/2022/8/narrativesnotime/1.0.0',
    #                'https://www.semanticweb/adiel/ontologies/2022/8/narrativesnomarg/1.0.0']
    # for i in range(len(list_of_IRI)):
    #     list_of_IRI[i] = '<' + list_of_IRI[i]
    list_of_codes = []
    # list_of_codes = [
    #     ['is_marginal', 'detection_of', 'serial_num', 'num_of_dets', 'lat', 'lon', 'detection', 'target', 'serial_num',
    #      'is_part_of_convoy', 'near_on_time', 'stop_on_time'], ['time', 'end_time']]
    list_of_IRI = ['<https://oxfordsemantic.tech/RDFox/getting-started#']  # schema-free-elements
    # list_of_codes.append(['poi', 'near_poi', 'poi_name'])  # schema-free-code

    # prefs = []
    # for ii, i in enumerate(list_of_IRI):
    #     for jj in list_of_codes[ii]:
    #         prefs.append("prefix " + jj + ": " + i + "> ")
    prefs = ['prefix : <https://oxfordsemantic.tech/RDFox/getting-started#>']
    is_rules = 1
    show_len_tar = 0

    beg1 = time.time()
    if to_ins == 1:
        if is_rules == 1:
            response1 = rule(
                '[?d, :is_marginal, "1"]:- [?d, :detection_of, ?t],[?d, :serial_num, ?n1],[?t, :num_of_dets, ?n2], FILTER(?n1=?n2),FILTER(?n2>1).')
            # print(response1.text)
            response2 = rule(
                '[?d, :is_marginal, "1"]:- [?d, :detection_of, ?t],[?d, :serial_num, ?n1],[?t, :num_of_dets, ?n2], FILTER(?n1=1),FILTER(?n2>1).')
            # print(response2.text)
            response3 = rule(
                '[?d1, :is_part_of_convoy, "1"],[?d2, :is_part_of_convoy, "1"]:-[?d1, :near_on_time, ?d2],[?d1, :near_on_time, ?d3],[?d1, :is_marginal, "0"],[?d2, :is_marginal, "0"],[?d3, :is_marginal, "0"],FILTER(?d2!=?d3).')
            # print(response3.text)
            response4 = rule(
                '[?d1, :stop_on_time, ?d2]:-[?d1, :near_on_time, ?d2],[?d1, :is_marginal, "1"],[?d2, :is_marginal, "1"].')
            # print(response4.text)
            # TODO(Natan) - This rule adds near_poi attribute to all targets which has near_poi relation
            response4 = rule(
                '[?t, :near_poi, ?p]:- [?d, :detection_of, ?t],[?d, :near_poi, ?p].')
            # print(response4.text)


        unique_times = np.unique(np.array(pd.read_csv(args.trajectoeis_csv).UTCTime))
        num_unique_trajectories = len(np.unique(np.array(pd.read_csv(args.trajectoeis_csv).TrajectoryID)))
        print(f"{num_unique_trajectories} Trajectories")
        df = pd.read_csv(args.poi_csv)
        poi_longs = np.array(df['long'])[:-2]
        poi_lats = np.array(df['lat'])[:-2]
        poi_name = np.array(df['name'])[:-2]
        # insert pois:
        for j in range(len(poi_longs)):
            # print(j)
            res = insert_data(f'"poi_{j}" a :poi ;\n    :lat   {poi_lats[j]}  ;\n    :lon  {poi_longs[j]} ;\n    :poi_name   "{poi_name[j]}"   .')
        # dpoi = query("SELECT ?p WHERE { ?p a :poi }")
        with open(args.trajectoeis_csv) as csvfile:
            all_hard0 = []
            timess = []
            timess11 = []
            timess22 = []
            targets = set()
            targets_lat_lon = defaultdict(list)
            for nc in range(num_of_chunks):
                beg = time.time()
                head = [next(csvfile) for x in range(chunk_size * nc, chunk_size * (nc + 1))]
                spamreader = csv.reader(head, delimiter=',', quotechar='/')
                for i, row in tqdm.tqdm(enumerate(spamreader), total=chunk_size):
                    detection_idx = nc * chunk_size + i
                    if i > 0:
                        if len(str(row[-1])) > 0:

                            r = row[5].split(' ')
                            uti = list(unique_times).index(row[5])
                            r1 = unique_times[uti + 1].split(' ')
                            t = query('SELECT ?n WHERE {"' + str(row[4]) + '" :num_of_dets ?n}')

                            if t.text == '?n\n':  # if its target id is  a new target
                                insert_data(f'{row[4]} a :target ;\n :num_of_dets 1 .')

                                res = insert_data('"' + str(detection_idx) + '" a :detection ;\n    :lat ' + row[
                                    8] + ';\n    :lon ' + row[7] + ';\n    :time "' + str(
                                    r[1]) + '";\n    :end_time "' + str(
                                    r1[1]) + '";\n    :serial_num 1;\n    :is_marginal "0" ;\n    :detection_of "' +
                                                  row[4] + '".')
                                # insert_data('"' + str(row[-1]) + '" a :detection ;\n    :lat ' + row[
                                #     8] + ';\n    :lon ' + row[7] + ';\n    :time "' + str(
                                #     r[1]) + '";\n    :end_time "' + str(r1[1]) + '";\n    :serial_num 1;\n    :is_marginal "0" ;\n    :detection_of "' +
                                #                   row[4] + '".')

                            else:

                                # datalog_text = "@PREFIX narr: <https://oxfordsemantic.tech/RDFox/getting-started/> . " \
                                #    '["'+row[4]+'", narr:num_of_dets, '+t.text.split('\n')[1]+'] .'
                                # response = requests.patch(rdfox_server1, params={"operation": "delete-content"}, data=datalog_text)
                                # response = delete_data(
                                #     '["' + row[4] + '", :num_of_dets, ' + t.text.split('\n')[1] + '] .')

                                # Natan code
                                num_of_dets = t.text.split('\n')[1]
                                deletion_query= f"""DELETE DATA
                                                {{
                                                  "{row[4]}"  :num_of_dets {num_of_dets} .
                                                }}"""
                                response = query(deletion_query)
                                response = insert_data('"' + row[4] + '" :num_of_dets ' + str(
                                    int(t.text.split('\n')[1]) + 1) + '.')

                                # d = '\n@prefix narr: <https://oxfordsemantic.tech/RDFox/getting-started/> .\n\n"'+row[4]+'" narr:num_of_dets '+str(int(t.text.split('\n')[1])+1)+'. \n \n'
                                # response = requests.post(rdfox_server1, data=d)
                                if show_len_tar == 1:
                                    ti = query('SELECT ?n WHERE {"' + row[4] + '" :num_of_dets ?n.}')
                                    # print('ttext_after_insert', ti.content)


                                res = insert_data('"' + str(detection_idx) + '" a :detection ;\n   :lat ' + row[
                                # res = insert_data('"' + str(row[-1]) + '" a :detection ;\n   :lat ' + row[
                                    8] + ';\n   :lon ' + row[7] + ';\n   :time "' + str(
                                    r[1]) + '";\n   :end_time "' + str(
                                    r1[1]) + '";\n   :serial_num ' + str(int(t.text.split('\n')[1]) + 1) + ';\n   :is_marginal "0" ;\n   :detection_of "' +
                                                  row[4] + '".')

                # new_tar = '\n@prefix narr: <https://oxfordsemantic.tech/RDFox/getting-started/> .\n\n"'+str(j)+'" a narr:poi ;\n   narr:lat '+str(poi_lats[j])+';\n   narr:lon '+str(poi_longs[j])+' . \n \n'
                # response = requests.post(rdfox_server1, data=new_tar)
                d = query(
                    "SELECT ?d ?lon ?lat ?t WHERE { ?d a :detection . ?d :lat ?lat .?d :lon ?lon. ?d :detection_of ?t} GROUP BY (?d)")
                dlist = d.text.split('\n')
                lats = []
                longs = []
                tars = []
                de = []
                for i in dlist[1:-1]:
                    pair = i.split('\t')
                    de.append(pair[0])
                    lats.append(float(pair[2]))
                    longs.append(float(pair[1]))
                    tars.append(pair[3])

                p = np.arange(num_of_poi)  # pois ids
                dis = np.zeros((len(lats), len(poi_lats)))
                for ii in range(len(lats)):
                    for jj in range(len(poi_lats)):
                        dis[ii, jj] = np.sqrt((lats[ii] - poi_lats[jj]) **
                                              2 + (longs[ii] - poi_longs[jj]) ** 2)
                # save inseces of pois and detections which are near each other
                coorsx, coorsy = np.where(dis < near_par)
                # insert between those couples a relation of "near_poi"
                for k in range(len(coorsx)):
                    res = insert_data(f'"{de[coorsx[k]]}"  :near_poi   "poi_{p[coorsy[k] - 1]}"  .')
                # dis=euclidean_distances
                # Natan assigned  neardets = 0 to save runtime for the demo
                neardets = 0
                if neardets == 1:
                    dis = np.ones((len(lats), len(lats)))
                    for ii in range(len(lats)):
                        # if np.mod(ii, 100) == 0:
                            # print(ii, len(lats))
                        # print(ii,len(lats))
                        for jj in range(ii, len(lats)):
                            if tars[ii] != tars[jj]:
                                dis[ii, jj] = np.sqrt((lats[ii] - lats[jj]) ** 2 + (longs[ii] - longs[jj]) ** 2)
                    coorsx, coorsy = np.where(dis < near_par)
                    for k in range(len(coorsx)):
                        if np.mod(k, 100) == 0:
                            if de[coorsx[k]] != de[coorsy[k]]:
                                response11 = query('INSERT {' + de[coorsx[k]] + ' :near_on_time ' + de[
                                    coorsy[k]] + '} WHERE {' + de[coorsx[k]] + ' :detection_of ?t1. ' + de[
                                                       coorsy[
                                                           k]] + ' :detection_of ?t2  . FILTER (?t1!=?t2).' +
                                                   de[coorsx[k]] + ' :time ?bt1. ' + de[
                                                       coorsy[k]] + ' :time ?bt2. ' + de[
                                                       coorsx[k]] + ' :end_time ?et1. ' + de[coorsy[
                                    k]] + ' :end_time ?et2. FILTER(?bt1<=?et2). FILTER(?bt2<=?et1)}')


                beg2 = time.time()
                time1 = beg2 - beg1

                print('time of insert ', time1)
                beg3 = time.time()
                time2 = beg3 - beg2
                print('time of commit ', time2)
                # insert rules:

                timess.append(time.time() - beg)
                # plt.figure(nc)
                #            img = cv2.imread(r"scaled2.jpg")

                #            plt.imshow(img, interpolation='sinc', extent=[34.84, 34.89, 32.3, 32.34])
                # plt.xlim([34.84, 34.89])
                # plt.ylim([32.3, 32.34])

                if to_ins == 0:
                    beg3 = time.time()
                # plot all detections
                # plot all pois
                d = query("SELECT ?p ?lon ?lat WHERE { ?p a :poi . ?p :lat ?lat .?p :lon ?lon. } GROUP BY{?p}")
                alon, alat = extract_coor(d)
                # print('##############', d.text, alat, alon)
                # plt.plot(alon, alat, 'b.')

                beg4 = time.time()
                time3 = beg4 - beg3
                print('time of extract detections ', time3)
                # Narrative 1: Locate all the detections that are near poi.
                if 0 in operated_narratives:
                    print('begins query 0')
                    d0 = query(
                        "SELECT ?d ?lon ?lat WHERE { ?d :detection_of ?p. ?d :lat ?lat. ?d :lon ?lon.} GROUP BY (?d)")
                    beg5 = time.time()
                    time4 = beg5 - beg4
                    alon0, alat0 = extract_coor(d0)
                    plt.plot(alon0, alat0, 'c.')
                    print('num of dets of tars', len(alat0))
                    print('time it took ', time4)

                if 1 in operated_narratives:
                    print('begins query 1')
                    d1 = query(
                        "SELECT ?d ?lon ?lat WHERE { ?d a :detection. ?d :near_poi ?p. ?d :lat ?lat. ?d :lon ?lon.} GROUP BY (?d)")
                    beg5 = time.time()
                    time4 = beg5 - beg4
                    alon1, alat1 = extract_coor(d1)
                    plt.plot(alon1, alat1, 'r.')
                    print('num of dets near poi', len(alat1))
                    print('time it took ', time4)
                # Narrative 2: list targets that has more than 'length' detections.
                if 2 in operated_narratives:
                    print('begins query 2')
                    i = 0
                    gg = query("SELECT ?t WHERE { ?t a :target . ?t :num_of_dets ?n .FILTER(?n>2).}")
                    beg6 = time.time()
                    time5 = beg6 - beg5
                    print('list of targets', gg)
                    print('time it took ', time5)

                # Narrative 3: plot detections of targets that pass near other targets.

                if 3 in operated_narratives:
                    print('begins query 3')
                    d3 = query(
                        "SELECT ?d ?lon ?lat WHERE { ?d :lat ?lat. ?d :lon ?lon. ?d :near_on_time ?d2.} GROUP BY (?d)")
                    blon3, blat3 = extract_coor(d3)
                    plt.plot(blon3, blat3, 'y.')
                    beg6 = time.time()
                    time5 = beg6 - beg5
                    print('num of detection in targets that pass through other targets', len(blon3))
                    print('time it took ', time5)
                # Narrative 3: plot detections of targets that atop near other targets.

                if 4 in operated_narratives:
                    print('begins query 4')
                    d4 = query(
                        "SELECT ?d ?lon ?lat WHERE { ?d :lat ?lat. ?d :lon ?lon. ?d :stop_on_time ?d2.}  GROUP BY (?d)")
                    blon4, blat4 = extract_coor(d4)
                    plt.plot(blon4, blat4, 'm.')
                    beg7 = time.time()
                    time6 = beg7 - beg6

                    print('num of detectios that stop next to other targets', len(blon4))
                    print('time it took ', time6)

                # Narrative 5: print all targets that carry at least consequentive 3 detections from convoys.
                if 5 in operated_narratives:
                    print('begins query 5')
                    d5 = query(
                        'SELECT ?t ?lon ?lat WHERE {?d :detection_of ?t.  ?d :is_part_of_convoy "1". ?d :lat ?lat. ?d :lon ?lon.}')
                    dat, _ = extract_data1(d5, 1)
                    t = dat[0]
                    lon = dat[1]
                    lat = dat[2]
                    beg8 = time.time()
                    time7 = beg8 - beg7
                    print('time it took ', time7)

                    lats = []
                    lons = []
                    trajs = []
                    print("extracting their targets")
                    for i, u in enumerate(t):
                        if t.count(u) > 1:
                            lats.append(float(lat[i]))
                            lons.append(float(lon[i]))
                            trajs.append(u)
                    plt.plot(lons, lats, 'g.', mfc='none')
                    print("num of convoys' detections", len(lats))
                    print("num of convoys", len(np.unique(trajs)))


                if 6 in operated_narratives:
                    print('begins query 6')
                    d4 = query(narrative_selection_query)
                    poi_name = extract_poi_name(narrative_selection_query)
                    tracks_near_poi = d4.text.split('\n')[1:]
                    targets_lat_lon = get_info_from_track_numbers(tracks_near_poi, targets_lat_lon)


                print('end chunk')
                timess22.append(time.time() - beg2)
                if is_prob == 1:
                    beg11 = time.time()

                    ruless = 'is_marginal(d,"1"):-detection_of(d,t),serial_num(d,n1),num_of_dets(t,n2).\nis_part_of_convoy(d1,"1"):-near_on_time(d1,d2),near_on_time(d1,d3),is_marginal(d1,"0"),is_marginal(d2,"0"),is_marginal(d3,"0").\nstop_on_time(d1,d2):-near_on_time(d1,d2),is_marginal(d1,"1"),is_marginal(d2,"1").\n'
                    d = query('SELECT ?o ?p ?s WHERE {?o ?p ?s}')
                    d6 = query('SELECT ?d ?p ?o WHERE {?d :is_part_of_convoy ?o. ?d  ?p ?o}')
                    d7 = query('SELECT ?d ?p ?o WHERE {?d :stop_on_time ?o. ?d  ?p ?o}')
                    # print('len init_probOD',len(init_probOD))
                    # print('len OD',len(OD))
                    # print('len pt',len(pt))
                    # print('len wheres',len(wheres))
                    list_of_pair = extract_data1(d)
                    list_of_pair6 = extract_data1(d6)
                    list_of_pair7 = extract_data1(d7)
                    all_inferred = list_of_pair6 + list_of_pair7
                    all_hard = []
                    for i in list_of_pair:
                        if i not in all_inferred:
                            all_hard.append(i)
                    iii = 0
                    p_hard = []
                    for i in all_hard:
                        if i in all_hard0:
                            iii = iii + 1
                            p_hard.append(former_p_hard[iii])
                        else:
                            p_hard.append(np.random.rand(1))
                    p_inffered = give_prob(p_hard, all_hard, all_inferred, ruless)
                    all_hard0 = all_hard
                    former_p_hard = p_hard
                    timess11.append(time.time() - beg11)

    res = query(f"""SELECT  ?lat ?lon 
                    WHERE {{
                      ?p a :poi .
                      ?p :poi_name "{poi_name}" ;
                       :lat ?lat  ;
                       :lon ?lon .
                    }}
        """
                )

    dlist = res.text.split('\n')
    for i in dlist[1:-1]:
        poi_lat, poi_lon = i.split('\t')


    poi_data = [poi_name, poi_lat, poi_lon]
    save_targets_lat_lon_to_visualizer_format(targets_lat_lon, poi_data)
    run_visualization_tool()

def run_visualization_tool():
    print("Running visualization tool")
    # Run the other script
    os.chdir('/home/natan/github/mapTool')
    subprocess.run(["/home/natan/projects/onto_llm/onto_llm_venv/bin/python", "/home/natan/github/mapTool/mainmap.py"])


def run_demo(args):
    narrative_selection_query = create_rules_from_prompt(args.ontology, args.rules)
    rdfox_server = "http://localhost:12110"
    working_dir = '.'
    with RDFOXApp(rdfox_executable_path=args.rdfox_exec,
                  data_store_name='narratives',
                  rdfox_server=rdfox_server,
                  working_dir=working_dir
                  ) as RdfoxContext:
        pass
        run_nerratives(args, rdfox_server, narrative_selection_query=narrative_selection_query)

def main():

    parser = argparse.ArgumentParser('VISDOM2D rdfox simulator')
    parser.add_argument("--rdfox_exec", help="Path to RDFOX executable", required=True)
    parser.add_argument("--rules", help="Path to rules", required=True)
    parser.add_argument("--ontology", help="Path to configurations", required=True)
    parser.add_argument("--poi_csv", help="Path to POI csv", required=True)
    parser.add_argument("--trajectoeis_csv", help="Path to trajectories csv", required=True)
    args = parser.parse_args()

    run_demo(args)

if __name__ == '__main__':
    main()
