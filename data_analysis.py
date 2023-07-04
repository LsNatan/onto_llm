import csv
from collections import defaultdict


import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import torch as torch
from tqdm import tqdm

from rdfox_app import RDFOXApp



class NarrativeRunner:
    def __init__(self, rdfox_server, datastore, poi_csv, trajectories_df):
        self.poi_csv = poi_csv
        self.trajectories_df = trajectories_df
        self.num_of_chunks = 2
        self.chunk_size = 2000  # at most 2000
        self.near_poi_const_in_km = 1


        self.rdfox_server1 = rdfox_server + "/datastores/" + datastore + "/content"
        self.rdfox_server2 = rdfox_server + "/datastores/" + datastore + "/sparql"
        self.list_of_IRI = ['<https://oxfordsemantic.tech/RDFox/getting-started#']
        self.prefs = ['prefix : <https://oxfordsemantic.tech/RDFox/getting-started#>']
        self.load_rules()
        self.load_pois()
        # self.get_poi_name_from_poi_id(dest_coord=(32.329, 34.8602))
        self.load_targets_detections()
        self.add_near_poi_attribute()

    def complex_rels_converter(self, data):
        if '{' in data:
            d = data.split('\n')
            for i in d:
                if '{' in i:
                    i = self.complex_rel_converter(i)
            converted_data = ''
            for i in d:
                converted_data = converted_data + i + '\n\n'
        else:
            converted_data = data
        return converted_data


    def complex_rel_converter(self, complex_rel):  # N3 syntax e.g. p:Bill cog:thinks {p:John f:loves p:Mary} .
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


    def insert_data(self, data):
        data = self.complex_rels_converter(data)
        st = ''
        for i in self.prefs:
            st = st + "\n@" + i + "."
        new_tar = st + '\n\n' + data + '\n \n'
        response = requests.post(self.rdfox_server1, data=new_tar)  # insert new target
        return response


    def delete_data(self, triples):
        st = ''
        for i in self.prefs:
            st = st + i + ". "
        datalog_text = st + " " \
                            ' ' + triples
        response = requests.patch(self.rdfox_server1, params={"operation": "delete-content"}, data=datalog_text)
        return response


    def query(self, query):
        st = ''
        for i in self.prefs:
            st = st + i
        sparql_text = st + \
                      query
        spl = query.split(' ')
        if spl[0][1:] == "ELECT":
            response = requests.get(self.rdfox_server2, params={"query": sparql_text})
        else:
            response = requests.get(self.rdfox_server2, params={"update": sparql_text})
        return response


    def rule(self, rule):
        st = ''
        for i in self.prefs:
            st = st + "\n@" + i + "."
        datalog_rule1 = st + \
                        rule
        response = requests.post(self.rdfox_server1, data=datalog_rule1)
        return response


    def extract_coor(self, d):
        dlist = d.text.split('\n')
        alon = []
        alat = []
        for i in dlist[1:-1]:
            pair = i.split('\t')
            alon.append(float(pair[1]))
            alat.append(float(pair[2]))
        return alon, alat


    def extract_data1(self, d, is_dat=0):
        dlist = d.text.split('\n')
        pairlen = len(dlist[0].split('\t'))
        list_of_pairs = []
        if is_dat == 1:
            dat = []
            for j in range(pairlen):
                dat.append([])
        for i in dlist[1:-1]:
            pair = i.split('\t')
            for i in self.list_of_IRI:
                pair[1] = pair[1].replace(i, '')
            pair[1] = pair[1].replace('>', '')
            for i in self.list_of_IRI:
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


    def load_rules(self):
        response1 = self.rule(
            '[?d, :is_marginal, "1"]:- [?d, :detection_of, ?t],[?d, :serial_num, ?n1],[?t, :num_of_dets, ?n2], FILTER(?n1=?n2),FILTER(?n2>1).')
        # print(response1.text)
        response2 = self.rule(
            '[?d, :is_marginal, "1"]:- [?d, :detection_of, ?t],[?d, :serial_num, ?n1],[?t, :num_of_dets, ?n2], FILTER(?n1=1),FILTER(?n2>1).')
        # print(response2.text)
        response3 = self.rule(
            '[?d1, :is_part_of_convoy, "1"],[?d2, :is_part_of_convoy, "1"]:-[?d1, :near_on_time, ?d2],[?d1, :near_on_time, ?d3],[?d1, :is_marginal, "0"],[?d2, :is_marginal, "0"],[?d3, :is_marginal, "0"],FILTER(?d2!=?d3).')
        # print(response3.text)
        response4 = self.rule(
            '[?d1, :stop_on_time, ?d2]:-[?d1, :near_on_time, ?d2],[?d1, :is_marginal, "1"],[?d2, :is_marginal, "1"].')
        # print(response4.text)
        # TODO(Natan) - This rule adds near_poi attribute to all targets which has near_poi relation
        response4 = self.rule(
            '[?t, :near_poi, ?p]:- [?d, :detection_of, ?t],[?d, :near_poi, ?p].')
        # print(response4.text)

    def load_pois(self):
        df = pd.read_csv(self.poi_csv)
        self.poi_lons = np.array(df['long'])[:-2]
        self.poi_lats = np.array(df['lat'])[:-2]
        self.poi_name = np.array(df['name'])[:-2]
        # insert pois:
        for j in range(len(self.poi_lons)):
            res = self.insert_data(f'"poi_{j}" a :poi ;\n    :lat   {self.poi_lats[j]}  ;\n    :lon  {self.poi_lons[j]} ;\n    :poi_name   "{self.poi_name[j]}"   .')


    def load_targets_detections(self):

        # Create an empty list to store the query strings
        full_query = """"""

        # Iterate over the rows of the DataFrame
        for i, row in tqdm(enumerate(self.trajectories_df.iterrows()), desc="Building insertion query", total=len(self.trajectories_df)):
            detection_id = str(row[1]['TrajectoryID'])
            lat = row[1]['GeoLocationY']
            lon = row[1]['GeoLocationX']

            full_query += f"""
"{i}" a :detection ;
    :lat {lat};
    :lon {lon};
    :detection_of "{detection_id}"."""
            # query_list.append(query)
        # print(full_query)
        res = self.insert_data(full_query)
    @staticmethod
    def calculate_distance_matrix_torch(lats, lons, poi_lats, poi_lons):
        # Convert degrees to radians and move tensors to GPU
        lats_rad = torch.deg2rad(torch.tensor(lats)).cuda()
        lons_rad = torch.deg2rad(torch.tensor(lons)).cuda()
        poi_lats_rad = torch.deg2rad(torch.tensor(poi_lats)).cuda()
        poi_lons_rad = torch.deg2rad(torch.tensor(poi_lons)).cuda()

        # Earth radius in kilometers
        earth_radius = 6371.0

        # Calculate differences in latitudes and longitudes
        delta_lats = lats_rad.unsqueeze(1) - poi_lats_rad
        delta_lons = lons_rad.unsqueeze(1) - poi_lons_rad

        # Haversine formula
        a = torch.sin(delta_lats / 2) ** 2 + torch.cos(lats_rad.unsqueeze(1)) * torch.cos(poi_lats_rad) * torch.sin(
            delta_lons / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance_matrix = earth_radius * c

        return distance_matrix.detach().cpu().numpy()


    @staticmethod
    def calculate_distance_matrix(lats, lons, poi_lats, poi_lons):
        # Convert degrees to radians
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        poi_lats_rad = np.radians(poi_lats)
        poi_lons_rad = np.radians(poi_lons)

        # Earth radius in kilometers
        earth_radius = 6371.0

        # Calculate differences in latitudes and longitudes
        delta_lats = lats_rad[:, np.newaxis] - poi_lats_rad
        delta_lons = lons_rad[:, np.newaxis] - poi_lons_rad

        # Haversine formula
        a = np.sin(delta_lats / 2) ** 2 + np.cos(lats_rad[:, np.newaxis]) * np.cos(poi_lats_rad) * np.sin(
            delta_lons / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance_matrix = earth_radius * c

        return distance_matrix

    def add_near_poi_attribute(self):
        lons = self.trajectories_df['GeoLocationX'].values.tolist()
        lats = self.trajectories_df['GeoLocationY'].values.tolist()
        print("Calculating distance matrix")
        # distance_matrix = self.calculate_distance_matrix_torch(lats, lons, [32.327], [34.862])
        # distance_matrix = self.calculate_distance_matrix(lats, lons, [32.327], [34.862])
        distance_matrix = self.calculate_distance_matrix(lats, lons, self.poi_lats, self.poi_lons)

        # save inseces of pois and detections which are near each other
        coorsx, coorsy = np.where(distance_matrix < self.near_poi_const_in_km)
        # insert between those couples a relation of "near_poi"
        if len(coorsx) > 0:
            for k in range(len(coorsx)):
                res = self.insert_data(f'"{coorsx[k]}"  :near_poi   "poi_{coorsy[k]}"  .')

    def get_poi_name_from_poi_id(self, dest_coord):
        lats, lons, = dest_coord
        distance_matrix = self.calculate_distance_matrix([lats], [lons], self.poi_lats, self.poi_lons)

        # save inseces of pois and detections which are near each other
        poi_id = np.argmin(distance_matrix)


        query = f"""SELECT ?poi_name  WHERE {{?p :poi_name ?poi_name . FILTER(?p = "poi_{poi_id}") }} """
        res = self.query(query)
        print(res.text.split('\n'))

    def query_for_targets_near_poi(self, narrative_selection_query):
        targets_lat_lon = defaultdict(list)
        res = self.query(narrative_selection_query)
        poi_name = self.extract_poi_name(narrative_selection_query)
        tracks_near_poi = res.text.split('\n')[1:]
        targets_lat_lon = self.get_info_from_track_numbers(tracks_near_poi, targets_lat_lon)

        res = self.query(f"""SELECT  ?lat ?lon 
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
        write_targets_to_vis_tool_csv(targets_lat_lon)


    @staticmethod
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


    def get_info_from_track_numbers(self, targets_near_poi, targets_lat_lon):
        for target in targets_near_poi:
            query_string = f"""SELECT DISTINCT  ?lat ?lon
                                WHERE {{
                                  ?d :detection_of ?t ;
                                     :lat ?lat ;
                                     :lon ?lon .
                                  FILTER(?t={target})
                                }}"""
            res = self.query(query_string)
            dlist = res.text.split('\n')
            for i in dlist[1:-1]:
                lat, lon = i.split('\t')
                targets_lat_lon[target].append((lat, lon))
        return targets_lat_lon
def extact_k_longest_targets(df, k):
    # Convert 'utctime' column to datetime
    df['datetime'] = pd.to_datetime(df['UTCTime'])

    # Group by 'ID' and calculate the duration
    grouped = df.groupby('TrajectoryID')['datetime'].apply(lambda x: x.max() - x.min())

    # Sort the durations in descending order
    sorted_durations = grouped.sort_values(ascending=False)

    # Get the top 10 IDs with the longest duration
    top_k_ids = sorted_durations.head(k)

    # Get the count of entries for each top 10 ID
    # entry_counts = df[df['TrajectoryID'].isin(top_k_ids.index)].groupby('TrajectoryID').size()

    # Print the result
    # print(entry_counts)

    print(list(dict(top_k_ids).keys()))
    return top_k_ids

def create_track_detection_dict(df, top_k_ids):
    # Filter the dataframe to include only the top 10 IDs
    filtered_df = df[df['TrajectoryID'].isin(top_k_ids.index)]

    # Create an empty dictionary to store the results
    result_dict = {}

    # Iterate over the top 10 IDs
    for id in top_k_ids.index:
        # Get the corresponding values in columns 'x' and 'y' for each entry
        values = filtered_df.loc[filtered_df['TrajectoryID'] == id, ['GeoLocationY', 'GeoLocationX']].values.tolist()
        # values = filtered_df.loc[filtered_df['TrajectoryID'] == id, ['GeoLocationY', 'GeoLocationX', 'datetime']].values.tolist()

        # Add the values to the dictionary
        result_dict[id] = values
    return result_dict

def write_targets_to_vis_tool_csv(track_detection_dict, poi_corrds):
    blue_contrast_palette = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 1.0, 0.0],  # Lime Green
        [0.0, 0.5, 1.0],  # Sky Blue
        [0.7, 0.0, 1.0],  # Violet
        [1.0, 0.0, 0.5]  # Rose
    ]

    header = 'r/d route order name  long  lat color'.split()
    order = 0
    poi_lat, poi_lon, = poi_corrds
    with open('longest_targets.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for i, (target, detections) in enumerate(track_detection_dict.items()):
            first_det_of_track = False
            color = str(blue_contrast_palette[i]).replace(',', "")
            for j, detection in enumerate(detections):
                lat, long = detection
                row = ['r', i, order, i if first_det_of_track else "", long, lat, color]
                # first_det_of_track = False
                # write the data
                writer.writerow(row)
                order += 1
        # # Write POI
        color = '[1 0 1]'
        row = ['d', i + 1, order, "Israel Post", poi_lon, poi_lat, color]
        writer.writerow(row)


def run_visualization_tool():
    print("Running visualization tool")
    # Run the other script
    os.chdir('/home/natan/github/mapTool')
    subprocess.run(["/home/natan/projects/onto_llm/onto_llm_venv/bin/python", "/home/natan/github/mapTool/mainmap.py"])

def plot_targets(track_detection_dict, poi):
    # Extract the latitude and longitude coordinates from result_dict
    coordinates = track_detection_dict.values()

    # Separate the latitude and longitude into separate lists
    # lats, lons = zip(*[coord for sublist in coordinates for coord in sublist])

    # Create a scatter plot of the coordinates
    for coord in coordinates:
        lats, lons = zip(*coord)
        plt.plot(lons, lats, color=np.random.random(3).tolist(), marker='o')

    # Set the x and y axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Set the plot title
    plt.title('Coordinates from result_dict')

    # Show the plot
    plt.show()


def build_kg_and_extract_relevant_data(df):
    rdfox_server = "http://localhost:12110"
    working_dir = '.'
    datastore = 'narratives'
    rdfox_exec = r"/home/natan/tools/RDFox-linux-x86_64-6.2/RDFox"
    with RDFOXApp(rdfox_executable_path=rdfox_exec,
                  data_store_name=datastore,
                  rdfox_server=rdfox_server,
                  working_dir=working_dir
                  ) as RdfoxContext:
        handler = NarrativeRunner(rdfox_server, datastore,
                               poi_csv=r"/home/natan/projects/onto_llm/data/POI_Netanya.csv",
                               trajectories_df=df
                               )

        narrative_selection_query =f""" SELECT ?t WHERE {{ ?d :detection_of ?t . ?d :near_poi ?p . ?p :poi_name "Israel Post" . }}
"""
        handler.query_for_targets_near_poi(narrative_selection_query)
def analyze_and_plot_data(df, k):
    top_k_ids = extact_k_longest_targets(df, k)
    track_detection_dict = create_track_detection_dict(df, top_k_ids)
    write_targets_to_vis_tool_csv(track_detection_dict, poi_corrds=[32.327, 34.862])
    # plot_targets(track_detection_dict)

def main():
    k = 10
    # trajectoeis_csv = r"/home/natan/Natanya_reduced.csv"
    trajectoeis_csv = r"/home/natan/Netanya_Trajectories.csv"
    df = pd.read_csv(trajectoeis_csv)
    analyze_and_plot_data(df, k)
    # build_kg_and_extract_relevant_data(df)
    run_visualization_tool()

if __name__ == '__main__':
    main()
