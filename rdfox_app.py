import copy
import re
import time

import requests
from prettytable import PrettyTable
from rdfox_runner import RDFoxRunner


class TimeMeasurement:
    # global_meas_dict = {}
    # global_default_dict = {}
    # basic_measurement_struct_factory = dict(durations=[], total_time=0, num_measurements=0, avg_time=0)
    def __init__(self, name, extra_fields = None):
        # self.name = name
        # self.extra_fields = extra_fields
        self.start = None
        self.end = None
        self.duration = None
        # if name not in self.global_meas_dict:
        #     self.init_global_meas_dicts()

    # def init_global_meas_dicts(self):
    #     # self.global_meas_dict[self.name] = dict(durations=[], total_time=0, num_measurements=0, avg_time=0).update(self.extra_fields)
    #     d = copy.deepcopy(self.basic_measurement_struct_factory)
    #     if self.extra_fields is not None:
    #         d.update(self.extra_fields)
    #     self.global_meas_dict[self.name] = d
    #     self.global_default_dict[self.name] = d



    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        # self.global_meas_dict[self.name]['durations'].append(duration)
        # self.global_meas_dict[self.name]['total_time'] += duration
        # self.global_meas_dict[self.name]['num_measurements'] += 1
        # self.global_meas_dict[self.name]['avg_time'] = self.global_meas_dict[self.name]['total_time'] /\
        #                                               self.global_meas_dict[self.name]['num_measurements']


class InsertionMeasurement:
    # cls_meas_dict = {}
    # basic_measurement_factory = dict(durations=[], total_time=0, num_measurements=0, avg_time=0,
    #                                  insertion_events=[], instances=[], fields_per_instance=[], bytes_per_instance=[])
    def __init__(self):
        self.name = 'Insertion'
        self.start = None
        self.end = None
        self.insertion_events = None
        self.instances = None
        self.fields_per_instance = None
        self.subject_characters_count = None
        self.total_instance_chars = None
        self.bytes_per_char = 1



        # self.init_global_meas_dicts()

    # def init_global_meas_dicts(self):
        # self.cls_meas_dict = copy.deepcopy(self.basic_measurement_factory)

    # @classmethod
    # def reset_cls_meas_dicts(cls):
    #     cls.cls_meas_dict = copy.deepcopy(cls.basic_measurement_factory)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        # self.cls_meas_dict['durations'].append(duration)
        # self.cls_meas_dict['total_time'] += duration
        # self.cls_meas_dict['num_measurements'] += 1
        # self.cls_meas_dict['avg_time'] = self.cls_meas_dict['total_time'] // self.cls_meas_dict['num_measurements']

    def gather_insertion_statistics(self, sparql_query):
        # 1. Number of Insertion events (occurrences of INSERT DATA)
        insertion_events = len(re.findall(r'INSERT\sDATA', sparql_query))

        # 2. Number of instances (number of dots coming after INSERT DATA)
        instances = len(re.findall(r'\s+\.\s*', sparql_query))


        total_instance_chars = sum([len(field) for field in re.findall(r'MeasurementsTracks:[a-zA-Z_]+\s+".*?"', sparql_query)])

        fields = re.findall(r'MeasurementsTracks:[a-zA-Z_]+\s+"(.*?)"', sparql_query)
        # 3. Number of fields per instance (number of properties)
        fields_per_instance = len(fields) // instances
        subject_characters_count = sum([len(field) for field in fields])
        characters_count = sum([len(field) for field in fields])

        self.insertion_events = insertion_events
        self.instances = instances
        self.fields_per_instance = fields_per_instance
        self.subject_characters_count = subject_characters_count * self.bytes_per_char
        self.total_instance_chars = total_instance_chars * self.bytes_per_char




class SelectionMeasurement:
    cls_meas_dict = {}
    basic_measurement_factory = dict(durations=[], total_time=0, num_measurements=0, avg_time=0, num_of_selected_entries=[])

    def __init__(self):
        self.name = 'Selection'
        self.start = None
        self.end = None
        self.duration = None
        self.num_of_selected_entries = None
        # self.init_global_meas_dicts()

    # def init_global_meas_dicts(self):
    #     self.cls_meas_dict = copy.deepcopy(self.basic_measurement_factory)
    #
    # @classmethod
    # def reset_global_meas_dicts(cls):
    #     cls.cls_meas_dict = copy.deepcopy(cls.basic_measurement_factory)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        # self.cls_meas_dict['durations'].append(duration)
        # self.cls_meas_dict['total_time'] += duration
        # self.cls_meas_dict['num_measurements'] += 1
        # self.cls_meas_dict['avg_time'] = self.cls_meas_dict['total_time'] // self.cls_meas_dict['num_measurements']

    def gather_selection_statistics(self, result):
        self.num_of_selected_entries = result
        # self.cls_meas_dict['num_of_selected_entries'].append(result)
        # print("Number of selected entries:", result)



class RDFOXApp:

    def __init__(self,
                 rdfox_executable_path: str = "",
                 data_store_name: str = "",
                 rules_path: str = "",
                 rdfox_server: str = "http://localhost:12110",
                 working_dir: str = ""
                 ):
        assert rdfox_executable_path != "", "Please provide rdfox_executable_path!"
        assert data_store_name != "", "Please provide data_store_name!"

        self.rdfox_executable_path = rdfox_executable_path
        self.data_store_name = data_store_name
        self.rules_path = rules_path
        self.rdfox_server = rdfox_server
        self.working_dir = working_dir
        self.insertion_stats_measurements = []
        self.selection_stats_measurements = []

    def reset_measurements(self):
        self.insertion_stats_measurements = []
        self.selection_stats_measurements = []
    def __enter__(self):
        input_files = {}
        script = [
            f'dstore create {self.data_store_name} type parallel-nn',
            f'active {self.data_store_name}',
            f'endpoint start',
        ]

        if self.rules_path:
            input_files = {"rules.dlog": self.rules_path}
            script.insert(1, 'import rules.dlog')


        self.rdfox_runner_context = RDFoxRunner(input_files=input_files,
                                                script=script,
                                                rdfox_executable=self.rdfox_executable_path,
                                                working_dir=self.working_dir
                                                )
        self.rdfox_runner_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rdfox_runner_context.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def assert_response_ok(response, message):
        if not response.ok:
            raise Exception(
                message + "\nStatus received={}\n{}".format(response.status_code, response.text))

    def insert(self, query):
        # Issue insert
        with InsertionMeasurement() as tm:
            response = requests.post(
                self.rdfox_server + f"/datastores/{self.data_store_name}/sparql", data={"update": query})
        tm.gather_insertion_statistics(query)
        with open('insertion_queries.txt', 'a') as f:
            f.write(query)

        self.insertion_stats_measurements.append(tm)
        self.assert_response_ok(response, "Failed to insert fact via sparql.")

    def select(self, query, print_results=True):
        with SelectionMeasurement() as tm:
            response = requests.get(
                self.rdfox_server + f"/datastores/{self.data_store_name}/sparql", params={"query": query})
        self.assert_response_ok(response, "Failed to run select query.")
        parsed_result, num_of_rows = self.parse_selection(response)
        tm.gather_selection_statistics(num_of_rows)
        self.selection_stats_measurements.append(tm)
        if print_results:
            print(parsed_result)


    def create_datastore(self):
        response = requests.post(self.rdfox_server + f"/datastores/{self.data_store_name}")
        self.assert_response_ok(response, "Failed delete datastore")

    def delete_datastore(self):
        response = requests.delete(self.rdfox_server + f"/datastores/{self.data_store_name}")
        self.assert_response_ok(response, "Failed delete datastore")

    @staticmethod
    def parse_selection(response):
        table = PrettyTable()
        add_header = True
        for line in response.text.split('\n'):
            if line:
                row = line.replace("?", "").split('\t')
                if add_header:
                    table.field_names = row
                    add_header = False
                else:
                    table.add_row(row)
        return table.get_string(), len(table.rows)


def main():
    data_store_name = "targets_ds"
    rdfox_server = "http://localhost:12110"
    rdfox_executable_path = r"C:\projects\AI\BrainBit\code\rdfox\RDFox-win64-x86_64-6.2\RDFox.exe"
    my_prefix = "PREFIX benchmark: <http://rdfox_benchmark.org/>"
    prefs = f""" 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        {my_prefix}
    """
    with RDFOXApp(rdfox_executable_path, data_store_name, rdfox_server) as app:
        time_duration_data = f"""{prefs}

        INSERT DATA {{
          _:target1_time_stamp_1 rdf:subject "target1" ;
              rdf:predicate "is a" ;
              rdf:object "cat" ;
              time:hasTemporalDuration [a time:TemporalDuration ;time:hasBeginning "2022-01-01T00:00:00"^^xsd:dateTime ;time:hasEnd "2022-01-14T23:59:59"^^xsd:dateTime      ] .
          _:target1_time_stamp_2 rdf:subject "target1" ;
              rdf:predicate "is a" ;
              rdf:object "dog" ;
              time:hasTemporalDuration [a time:TemporalDuration ;time:hasBeginning "2022-01-15T00:00:00"^^xsd:dateTime ;time:hasEnd "2022-01-28T23:59:59"^^xsd:dateTime      ] .
          _:target1_time_stamp_3 rdf:subject "target1" ;
              rdf:predicate "is a" ;
              rdf:object "cat" ;
              time:hasTemporalDuration [a time:TemporalDuration ;time:hasBeginning "2022-01-29T00:00:00"^^xsd:dateTime ;time:hasEnd "2022-02-11T23:59:59"^^xsd:dateTime      ] .
        }}
        """

        time_duration_data1 = f"""{prefs}

            INSERT DATA {{
              benchmark:target1 rdf:type benchmark:target ;
                benchmark:classifiedAs [
                  benchmark:classification "tank" ;
                  benchmark:fromTime 0 ;
                  benchmark:toTime 100 ;
                  benchmark:prob 0.3
                ], [
                  benchmark:classification "suv" ;
                  benchmark:fromTime 100 ;
                  benchmark:toTime 200 ;
                  benchmark:prob 0.5
                ], [
                  benchmark:classification "bus" ;
                  benchmark:fromTime 200 ;
                  benchmark:toTime 300 ;
                  benchmark:prob 0.6
                ], [
                  benchmark:classification "tank" ;
                  benchmark:fromTime 300 ;
                  benchmark:toTime "now" ;
                  benchmark:prob 0.9
                ] .

              benchmark:target2 rdf:type benchmark:target ;
                benchmark:classifiedAs [
                  benchmark:classification "cat" ;
                  benchmark:fromTime 0 ;
                  benchmark:toTime 100 ;
                  benchmark:prob 0.1
                ], [
                  benchmark:classification "dog" ;
                  benchmark:fromTime 100 ;
                  benchmark:toTime 200 ;
                  benchmark:prob 0.2
                ], [
                  benchmark:classification "rat" ;
                  benchmark:fromTime 200 ;
                  benchmark:toTime 300 ;
                  benchmark:prob 0.6
                ], [
                  benchmark:classification "cat" ;
                  benchmark:fromTime 300 ;
                  benchmark:toTime "now" ;
                  benchmark:prob 0.9
                ] .

        }}
        """

        time_duration_data2 = f"""{prefs}
            INSERT DATA {{
              "windowed_event_1" a benchmark:window_event ;
                  benchmark:Target "target1" ;
                  benchmark:Classification "tank" ;
                  benchmark:FromTime 0 ;
                  benchmark:ToTime 100 ;
                  benchmark:Prob 0.7 .

              "windowed_event_2" a benchmark:window_event ;
                  benchmark:Target "target1" ;
                  benchmark:Classification "bus" ;
                  benchmark:FromTime 100 ;
                  benchmark:ToTime 200 ;
                  benchmark:Prob 0.3 .

              "windowed_event_3" a benchmark:window_event ;
                  benchmark:Target "target1" ;
                  benchmark:Classification "suv" ;
                  benchmark:FromTime 200 ;
                  benchmark:ToTime 300 ;
                  benchmark:Prob 0.3 .

              "windowed_event_4" a benchmark:window_event ;
                  benchmark:Target "target1" ;
                  benchmark:Classification "tank" ;
                  benchmark:FromTime 300 ;
                  benchmark:ToTime 'now' ;
                  benchmark:Prob 0.9 .  

              "windowed_event_5" a benchmark:window_event ;
                  benchmark:Target "target2" ;
                  benchmark:Classification "fighter" ;
                  benchmark:FromTime 15 ;
                  benchmark:ToTime 100 ;
                  benchmark:Prob 0.8 .

              "windowed_event_6" a benchmark:window_event ;
                  benchmark:Target "target2" ;
                  benchmark:Classification "cesna" ;
                  benchmark:FromTime 100 ;
                  benchmark:ToTime 200 ;
                  benchmark:Prob 0.3 .

              "windowed_event_7" a benchmark:window_event ;
                  benchmark:Target "target2" ;
                  benchmark:Classification "superman" ;
                  benchmark:FromTime 200 ;
                  benchmark:ToTime 315 ;
                  benchmark:Prob 0.4 .

              "windowed_event_8" a benchmark:window_event ;
                  benchmark:Target "target2" ;
                  benchmark:Classification "fighter" ;
                  benchmark:FromTime 315 ;
                  benchmark:ToTime 100 ;
                  benchmark:Prob 0.96 .                                                                
        }}
        """

        time_duration_select_query = f"""{prefs}
        SELECT ?obj ?begin WHERE {{
          ?ts benchmark:target1 ;
             rdf:object ?obj ;
             time:hasTemporalDuration ?t .
          ?t time:hasBeginning ?begin ;
             time:hasEnd ?end .
          FILTER("2022-01-29T00:00:00"^^xsd:dateTime <= ?begin && "2022-02-12T23:59:59"^^xsd:dateTime >= ?end)
        }}
        """

        time_duration_select_query1 = f"""{prefs}
        SELECT ?target ?classification
        WHERE {{

            ?target rdf:type benchmark:target .
            ?target benchmark:classifiedAs ?class .
            ?class benchmark:classification ?classification ;
                  benchmark:fromTime ?from ;
                  benchmark:toTime ?to .
            FILTER(?from <= 250 && ?to > 250)
            }}
        """

        time_duration_select_query2 = f"""{prefs}
        SELECT ?target ?classification
        WHERE {{
            ?event a benchmark:window_event ;
                   benchmark:Target ?target ;
                   benchmark:Classification ?classification ;
                   benchmark:FromTime ?from ;
                   benchmark:Prob ?prob ; 
                   benchmark:ToTime ?to .

            FILTER(?from <= 300 && ?to > 200)
        }}
        """

        app.insert(time_duration_data2)
        app.select(time_duration_select_query2)


if __name__ == '__main__':

    for _ in range(1):
        main()