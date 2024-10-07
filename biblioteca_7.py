#!/usr/bin/env python3
''' Mavlink telemetry (.tlog) file parser.

    Operates as a generator. Allows csv output or listing useful types/fields.
'''

import json
from pathlib import Path
from fnmatch import fnmatch
from pymavlink import mavutil
from inspect import getfullargspec
from datetime import datetime
from datetime import timezone
import numpy as np


class Telemetry:
    DEFAULT_FIELDS = {
        'VFR_HUD': ['heading', 'alt', 'climb', 'groundspeed', 'airspeed', "throttle"],
        #'VIBRATION': [f'vibration_{axis}' for axis in 'xyz'],
        #'SCALED_IMU2': [s for axis in 'xyz' for s in (axis + 'acc', axis + 'gyro')],
        #'ATTITUDE': [s for rot in ('roll', 'pitch', 'yaw') for s in (rot, rot + 'speed')],
        'GLOBAL_POSITION_INT': ['vx'],  # Added 'vx' for forward speed
        #'SCALED_PRESSURE2': ['temperature'],
        'GPS2_RAW': ['eph'],
        'GPS_RAW_INT': ['eph']
    }

    ANOMALY_THRESHOLDS = {
        #variação de velocidade??
        # SPEED AIRSPEED_AUTOCAL
        'AIRSPEED_AUTOCAL.vx': (-39.098, 39.098),  #x-direction (m/s) -29.15 33.16 (mean 0.08)
        'AIRSPEED_AUTOCAL.vy': (-39.098, 39.098),  #y-direction (m/s) -28.26 33.19 mean 0.01
        'AIRSPEED_AUTOCAL.vz': (-39.098, 39.098),  #z-direction (m/s) -5.69 5.88 mean 0

        # SPEED GLOBAL_POSITION_INT
        'GLOBAL_POSITION_INT.vx': (-3909.8, 3909.8),  #x-direction (cm/s) -2946 3312 mean 5.81
        'GLOBAL_POSITION_INT.vy': (-3909.8, 3909.8),  #y-direction (cm/s) -2841 3332 mean 1.35
        'GLOBAL_POSITION_INT.vz': (-3909.8, 3909.8),  #z-direction (cm/s) -670 577 mean 0.45

        # SPEED LOCAL_POSITION_NED
        'LOCAL_POSITION_NED.vx': (-39.098, 39.098),  #x-direction (m/s) -29.47 33.13 mean 0.06
        'LOCAL_POSITION_NED.vy': (-39.098, 39.098),  #y-direction (m/s) -28.42 33.33 mean 0.02
        'LOCAL_POSITION_NED.vz': (-39.098, 39.098),  #z-direction (m/s) -6.71 5.78 mean 0

        # SPEED GPS2_RAW and GPS_RAW_INT
        'GPS2_RAW.vel': (2160.6, 3909.8),  #Threshold for GPS-derived groundspeed (cm/s) similar to: GPS_RAW_INT.vel and VFR_HUD.groundspeed
        'GPS_RAW_INT.vel': (2160.6, 3909.8),  #Threshold for GPS-derived groundspeed (cm/s) similar to: GPS2_RAW.vel and VFR_HUD.groundspeed

        # SPEED VFR_HUD
        'VFR_HUD.airspeed': (21.606, 39.098),  #42 and 76 knots 21.606, 39.098 (m/s)
        'VFR_HUD.groundspeed': (21.606, 39.098),  #(m/s) similar to: GPS_RAW_INT.vel and GPS2_RAW.vel

        # Accelerometers RAW_IMU
        #'RAW_IMU.xacc': (0, 3.8),  # Threshold x-axis (0 to 3.8 g) verificar unidades??? -11500 e 500
        #'RAW_IMU.yacc': (0, 3.8),  # Threshold y-axis (0 to 3.8 g) -1000 e 3000
        #'RAW_IMU.zacc': (0, 3.8),  # Threshold z-axis (0 to 3.8 g) -1600 e 200

        # Accelerometers SCALED_IMU2
        #'SCALED_IMU2.xacc': (0, 3.8),  # Threshold x-axis (0 to 3.8 g) (mG) valores -380 a 501 (spike no final de -11000)
        #'SCALED_IMU2.yacc': (0, 3.8),  # Threshold y-axis (0 to 3.8 g) (mG) -933 a 1177  (spike no final de 3333)
        #'SCALED_IMU2.zacc': (0, 3.8),  # Threshold z-axis (0 to 3.8 g) (mG) -1533 a -361

        # ALTITUDE
        'VFR_HUD.alt': (0, 3078),  # Altitude between 0 and 3078 meters (SIMILAR A AHRS3.altitude)
        'AHRS3.altitude': (0, 3078),  # Absolute altitude in meters (SIMILAR A VFR_HUD.alt)
        'GPS_RAW_INT.alt': (0, 3078000),  # Altitude in milimeters (SIMILAR A GPS2_RAW)
        'GPS2_RAW.alt': (0, 3078000),  # Absolute altitude in milimeters (SIMILAR A GPS_RAW_INT)
        'GLOBAL_POSITION_INT.alt': (0, 3078000),  # Altitude in milimeters
        'GLOBAL_POSITION_INT.relative_alt': (0, 3078000),  # Relative altitude in milimeters
        'LOCAL_POSITION_NED.z': (-3078, 0),  # Altitude in local frame

        # HDOP (Horizontal Dilution of Precision)
        'GPS2_RAW.eph': (0, 10),  # GPS2_RAW eph gps2 raw_t(Min:60 Max: 139 Mean: 75.83)
        'GPS_RAW_INT.eph': (0, 10),  # GPS_RAW_INT eph gps raw int t(Min:60 Max:142 Mean:75.42)

        #visible satellites
        'GPS2_RAW.satellites_visible': (5, 20),  # Number of satellites visible in GPS2_RAW (0 to 20)
        'GPS_RAW_INT.satellites_visible': (5, 20),  # Number of satellites visible in GPS_RAW_INT (0 to 20)

        #Temperaturas
        'SCALED_PRESSURE.temperature': (4000, 5600),  # cdegC similares (3998; 5558) 4293.4 equivale em Cº (39,98; 55,58) 42,934
        'SENSOR_OFFSETS.raw_temp': (4000, 5600),  # cdegC similares (4000; 5558) media: 4293.18
    }

    similar_pairs = {
	    ##velocity
        ('GPS2_RAW.vel', 'GPS_RAW_INT.vel'): 'GPS-derived groundspeed comparison (cm/s)',
        ('GPS2_RAW.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison (m/s)',
        ('GPS_RAW_INT.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison (m/s)',
        ##altitude
        #('VFR_HUD.alt', 'AHRS3.altitude'): 'Altitude comparison (meters)',
        #('GPS_RAW_INT.alt', 'GPS2_RAW.alt'): 'GPS RAW vs GPS2 RAW altitude comparison (millimeters)',
        #('VFR_HUD.alt', 'GPS_RAW_INT.alt'): 'VFR HUD vs GPS RAW altitude comparison (meters to millimeters)',
        #('VFR_HUD.alt', 'GPS2_RAW.alt'): 'VFR HUD vs GPS2 RAW altitude comparison (meters to millimeters)',
        #('AHRS3.altitude', 'GPS_RAW_INT.alt'): 'AHRS3 vs GPS RAW altitude comparison (meters to millimeters)',
        #('AHRS3.altitude', 'GPS2_RAW.alt'): 'AHRS3 vs GPS2 RAW altitude comparison (meters to millimeters)',
        ##HDOP
        #('GPS2_RAW.eph', 'GPS_RAW_INT.eph'): 'GPS2 RAW vs GPS RAW HDOP comparison',
        ##satellites
        #('GPS2_RAW.satellites_visible', 'GPS_RAW_INT.satellites_visible'): 'GPS2 RAW vs GPS RAW visible satellites comparison'
    }

    def compare_similar_pairs(self, data_list):
        ''' Compares similar field pairs across multiple data entries and returns statistical results. '''
        results = {}

        print(f"Total messages collected for comparison: {len(data_list)}")

        for (field1, field2), description in self.similar_pairs.items():
            differences = []
            print(f"Comparing {field1} with {field2}")  # Add a debug statement
            for data in data_list:
                value1 = self.get_value_from_data(data, field1)
                value2 = self.get_value_from_data(data, field2)

                # Log the values before comparing
                print(f"Values for {field1} and {field2}: {value1}, {value2}")

                if value1 is None or value2 is None:
                    print(f"Skipping {description} due to missing values: {value1}, {value2}")
                    continue

                # Skip if either value is missing
                if value1 is None or value2 is None:
                    print(f"Skipping {description} due to missing values: {value1}, {value2}")
                    continue

                # Convert units if necessary (e.g., mm to m)
                if 'alt' in field1 or 'alt' in field2:
                    # Convert millimeters to meters if needed
                    print(f"Before conversion: {field1} = {value1}, {field2} = {value2}")
                    value1, value2 = self.unit_conversion(field1, value1, field2, value2)
                    print(f"After conversion: {field1} = {value1}, {field2} = {value2}")

                try:
                    diff = float(value1) - float(value2)
                    differences.append(diff)
                except ValueError:
                    print(f"Skipping non-numeric comparison for {field1} and {field2}: {value1}, {value2}")
                    pass  # Ignore non-numeric values

            if differences:
                results[description] = {
                    'highest': np.max(differences),
                    'lowest': np.min(differences),
                    'average': np.mean(differences)
                }
            else:
                print(f"No valid comparisons for {description}")

        return results

    def unit_conversion(self, field1, value1, field2, value2):
        ''' Converts fields to the same units if necessary (e.g., meters to millimeters or vice versa). '''
        # Convert millimeters to meters for altitude fields
        if ('alt' in field1 and 'alt' in field2):
            if 'mm' in field1:
                value1 /= 1000  # Convert mm to meters
            if 'mm' in field2:
                value2 /= 1000  # Convert mm to meters
        return value1, value2

    def get_value_from_data(self, data, field):
        ''' Helper to extract a nested value (e.g., "GPS_RAW_INT.vel") from data dict '''
        message_type, attr = field.split('.')
        if message_type in data:
            if attr in data[message_type]:
                return data[message_type][attr]
            else:
                 print(f"Field {attr} not found in message {message_type}. Available fields: {list(data[message_type].keys())}")
        else:
            print(f"Message type {message_type} not found in data. Available message types: {list(data.keys())}")
        return None
    
    def _update(self, type_, data, convert=lambda d: d):
        ''' Update with the latest data for 'type_'. '''
        if type_ not in self.offsets:
            return  # Skip if type is not in offsets

        offset = self.offsets[type_]

        # Ensure data list is large enough to accommodate new fields
        expected_size = offset + len(self.fields[type_])
        if len(self.data) < expected_size:
            # Extend self.data with NaN or a default value if the list is too short
            self.data.extend([float('nan')] * (expected_size - len(self.data)))

        for index, desired_attr in enumerate(self.fields.get(type_, [])):
            # Safely access data to avoid missing fields
            if desired_attr in data:
                try:
                    self.data[offset + index] = convert(data[desired_attr])
                except IndexError:
                    print(f"Error: Index out of range for field {desired_attr} in message type {type_}.")
                    continue
            else:
                print(f"Warning: Field {desired_attr} not found in message type {type_}.")
    '''
    def __init__(self, log_file, fields=DEFAULT_FIELDS, dialect='ardupilotmega'):
        #Creates a tlog parser on 'log_file', extracting 'fields'.
        self.log_file = str(log_file)  # mavutil doesn't use Path
        self.mlog = mavutil.mavlink_connection(self.log_file, dialect=dialect)
        self._init_fields(fields)
    '''

    def __init__(self, log_file, fields=DEFAULT_FIELDS, dialect='ardupilotmega'):
        ''' Creates a tlog parser on 'log_file', extracting 'fields'. '''
        self.log_file = str(log_file)  # mavutil doesn't use Path

        # Progress callback function with only new_pct argument
        def progress_callback(new_pct):
            print(f"Progress: {new_pct:.2%}")

        # Adding the progress callback
        self.mlog = mavutil.mavlink_connection(self.log_file, dialect=dialect, progress_callback=progress_callback)
        self._init_fields(fields)



    def _init_fields(self, fields):
        ''' Determine CSV fields and populate None attribute lists. '''
        if isinstance(fields, (str, Path)):
            with open(fields) as field_file:
                fields = json.load(field_file)

        self.csv_fields = ['timestamp', 'readable_timestamp']  # Add human-readable timestamp
        nan = float('nan')  # start with non-number data values
        self.data = [nan]
        self.offsets = {}
        for type_, field in fields.items():
            if field is None:
                type_class = f'MAVLink_{type_.lower()}_message'
                fields[type_] = getfullargspec(getattr(mavutil.mavlink, type_class).__init__).args[1:]
            self.offsets[type_] = offset = len(self.csv_fields)
            self.csv_fields.extend(f'{type_}.{attr}' for attr in fields[type_])
            self.data.extend(nan for _ in range(len(self.csv_fields) - offset))

        self.fields = fields
        self.type_set = set(fields)  # put major fields in a set
        self.csv_fields.append('Anomalies')  # Add anomalies column

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.mlog.close()

    def __iter__(self):
        ''' Iterate through available messages. '''
        while msg := self.mlog.recv_match(type=self.type_set):
            yield msg

    @staticmethod
    def match_types(m_type, patterns):
        ''' Return True if m_type matches one of patterns. '''
        return any(fnmatch(m_type, p) for p in patterns)

    def check_for_anomalies(self, msg_type, data):
        ''' Checks the current message for anomalies based on predefined thresholds. '''
        anomalies = []
        for field_name, (lower, upper) in self.ANOMALY_THRESHOLDS.items():
            anomaly_msg_type, field = field_name.split('.')
            if anomaly_msg_type == msg_type:
                value = data.get(field)
                if value is not None:
                    try:
                        value = float(value)
                        if not (lower <= value <= upper):
                            anomalies.append(f'{field_name}: {value}')
                    except ValueError:
                        pass  # Skip non-numeric values
        return anomalies

    def to_csv(self, output=None, anomalies_output=None, csv_sep=',', verbose=True):
        ''' Converts the parsed telemetry log into two CSV files:
            1. One with default fields (without anomalies)
            2. Another with timestamp, readable timestamp, and detected anomalies
            (up to 5 anomalies in separate columns)
        '''
        if output is None:
            output = Path(self.log_file).with_suffix('.csv')
        if anomalies_output is None:
            anomalies_output = Path(self.log_file).with_suffix('.anomalies.csv')

        if verbose:
            print(f'Processing {self.log_file}\n  -> Saving fields to {output}\n  -> Saving anomalies to {anomalies_output}')

        last_timestamp = None
        adding_fields = Path(output).is_file()
        adding_anomalies = Path(anomalies_output).is_file()

        with self as mavlink, open(output, 'a') as out_file, open(anomalies_output, 'a') as anomaly_file:
            def write_line(data, file):
                print(csv_sep.join(data), file=file)

            # Prepare CSV header for both files
            self.data = [str(val) for val in self.data]
            if not adding_fields:
                write_line(self.csv_fields[:-1], out_file)  # Write header for main CSV (fields), excluding the anomalies column
            if not adding_anomalies:
                write_line(['timestamp', 'readable_timestamp', 'anomaly_1', 'anomaly_2', 'anomaly_3', 'anomaly_4', 'anomaly_5'], anomaly_file)  # Write header for anomalies CSV

            for msg in mavlink:
                timestamp = getattr(msg, '_timestamp', 0.0)
                data = msg.to_dict()
                msg_type = msg.get_type()

                print(f"Message Type: {msg_type}, Fields: {list(data.keys())}")
                # Use a timezone-aware object for UTC
                readable_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Update the data with the current message
                self._update(msg_type, data, convert=str)

                # Check for anomalies in the current message
                anomalies = self.check_for_anomalies(msg_type, data)

                # Write data to CSV when timestamp changes
                if last_timestamp is not None and timestamp != last_timestamp:
                    self.data[0] = f'{last_timestamp:.8f}'  # Original timestamp
                    self.data[1] = readable_timestamp  # Human-readable timestamp

                    # Write the main data CSV (excluding the anomalies column)
                    write_line(self.data[:-1], out_file)

                    # Write anomalies in separate columns (up to 5 anomalies)
                    if anomalies:
                        anomaly_row = [f'{last_timestamp:.8f}', readable_timestamp] + anomalies[:5] + [''] * (5 - len(anomalies))
                        write_line(anomaly_row, anomaly_file)

                last_timestamp = timestamp

            # Write the last line of data
            if last_timestamp is not None:
                readable_timestamp = datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                self.data[0] = f'{last_timestamp:.8f}'
                self.data[1] = readable_timestamp

                # Write the main data CSV (excluding the anomalies column)
                write_line(self.data[:-1], out_file)

                # Write anomalies in separate columns (up to 5 anomalies)
                if anomalies:
                    anomaly_row = [f'{last_timestamp:.8f}', readable_timestamp] + anomalies[:5] + [''] * (5 - len(anomalies))
                    write_line(anomaly_row, anomaly_file)
            else:
                print('No desired messages found in file')

    def to_csv(self, output=None, anomalies_output=None, comparisons_output=None, csv_sep=',', verbose=True):
        ''' Converts the parsed telemetry log into CSV files:
            1. One with default fields (without anomalies)
            2. Another with detected anomalies
            3. Another with comparisons between similar fields
        '''
        if output is None:
            output = Path(self.log_file).with_suffix('.csv')
        if anomalies_output is None:
            anomalies_output = Path(self.log_file).with_suffix('.anomalies.csv')
        if comparisons_output is None:
            comparisons_output = Path(self.log_file).with_suffix('.comparisons.csv')

        if verbose:
            print(f'Processing {self.log_file}\n  -> Saving fields to {output}\n  -> Saving anomalies to {anomalies_output}\n  -> Saving comparisons to {comparisons_output}')

        last_timestamp = None
        adding_fields = Path(output).is_file()
        adding_anomalies = Path(anomalies_output).is_file()

        with self as mavlink, open(output, 'a') as out_file, open(anomalies_output, 'a') as anomaly_file, open(comparisons_output, 'a') as comparison_file:
            def write_line(data, file):
                print(csv_sep.join(data), file=file)

            # Prepare CSV headers
            self.data = [str(val) for val in self.data]
            if not adding_fields:
                write_line(self.csv_fields[:-1], out_file)  # Excluding anomalies column for the main CSV
            if not adding_anomalies:
                write_line(['timestamp', 'readable_timestamp', 'anomaly_1', 'anomaly_2', 'anomaly_3', 'anomaly_4', 'anomaly_5'], anomaly_file)
            if not Path(comparisons_output).is_file():
                write_line(['Description', 'Highest Difference', 'Lowest Difference', 'Average Difference'], comparison_file)

            # Store data for comparisons
            data_list = []

            for msg in mavlink:
                timestamp = getattr(msg, '_timestamp', 0.0)
                data = msg.to_dict()
                msg_type = msg.get_type()

                print(f"Message Type: {msg_type}, Fields: {list(data.keys())}")

                readable_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Update the data with the current message
                self._update(msg_type, data, convert=str)

                # Check for anomalies in the current message
                anomalies = self.check_for_anomalies(msg_type, data)

                # Collect data for later comparison
                data_list.append(data)

                if len(data_list) % 100 == 0:  # Print a sample every 100 messages
                    print(f"Collected {len(data_list)} messages. Sample: {data_list[-1]}")
                # Write to the main CSV if timestamp changes
                if last_timestamp is not None and timestamp != last_timestamp:
                    self.data[0] = f'{last_timestamp:.8f}'  # Original timestamp
                    self.data[1] = readable_timestamp  # Human-readable timestamp
                    write_line(self.data[:-1], out_file)

                    # Write anomalies
                    if anomalies:
                        anomaly_row = [f'{last_timestamp:.8f}', readable_timestamp] + anomalies[:5] + [''] * (5 - len(anomalies))
                        write_line(anomaly_row, anomaly_file)

                last_timestamp = timestamp

            # Write the last line of data
            if last_timestamp is not None:
                readable_timestamp = datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                self.data[0] = f'{last_timestamp:.8f}'
                self.data[1] = readable_timestamp
                write_line(self.data[:-1], out_file)

                if anomalies:
                    anomaly_row = [f'{last_timestamp:.8f}', readable_timestamp] + anomalies[:5] + [''] * (5 - len(anomalies))
                    write_line(anomaly_row, anomaly_file)
            else:
                print('No desired messages found in file')

            # Perform comparisons on data_list
            # Perform comparisons on data_list
            comparison_results = self.compare_similar_pairs(data_list)
            if comparison_results:
                print(f"Writing {len(comparison_results)} comparisons to file.")
                for description, stats in comparison_results.items():
                    print(f"Comparison result for {description}: Highest = {stats['highest']}, Lowest = {stats['lowest']}, Average = {stats['average']}")
                    comparison_row = [description, f"{stats['highest']:.6f}", f"{stats['lowest']:.6f}", f"{stats['average']:.6f}"]
                    write_line(comparison_row, comparison_file)
            else:
                print("No comparisons to write to file.")


    @classmethod
    def logs_to_csv(cls, output, logs, fields=DEFAULT_FIELDS, csv_sep=',', dialect='ardupilotmega', verbose=True):
        for log in logs:
            cls(log, fields, dialect).to_csv(output, csv_sep, verbose)

    @staticmethod
    def csv_to_df(filename, timestamp='timestamp', timezone='Australia/Melbourne', **kwargs):
        ''' Returns a pandas dataframe of a csv-log, indexed by timestamp. '''
        import pandas as pd

        def parser(utc_epoch_seconds):
            return pd.to_datetime(utc_epoch_seconds, unit='s').tz_localize('utc').tz_convert(timezone)

        return pd.read_csv(filename, index_col=timestamp, parse_dates=[timestamp], date_parser=parser, **kwargs)

    @classmethod
    def get_useful_fields(cls, tlogs, out='useful.json', fields=None, dialect='ardupilotmega', verbose=True):
        ''' Returns a {type: [fields]} dictionary of all non-constant fields. '''
        mavutil.set_dialect(dialect)
        fields = cls.__create_field_tracker(fields)
        init_fields = {type_: list(fields_) for type_, fields_ in fields.items()}
        useful_types = {}

        for tlog in tlogs:
            if verbose:
                print(f'Extracting useful fields from {tlog!r}')
            with cls(tlog, init_fields, dialect) as mavlink:
                for msg in mavlink:
                    cls.__process(msg, mavlink, fields, init_fields, useful_types)

        useful_types = {t: useful_types[t] for t in sorted(useful_types)}

        if out:
            with open(out, 'w') as output:
                json.dump(useful_types, output, indent=4)
            if verbose:
                print(f'  -> Saving to {output}')

        return useful_types

    @staticmethod
    def __create_field_tracker(fields):
        ''' Create a dictionary of {type: {field: None}} for specified fields. '''
        def get_fields(type_):
            return {field: None for field in getfullargspec(getattr(mavutil.mavlink, type_).__init__).args[1:]}

        if fields is None:
            fields = {t[8:-8].upper(): get_fields(t) for t in dir(mavutil.mavlink) if t.startswith('MAVLink_') and t.endswith('_message')}
        elif isinstance(fields, (str, Path)):
            fmt = 'MAVLink_{}_message'
            with open(fields) as in_file:
                fields = {type_: ({field: None for field in fields_} if fields_ else get_fields(fmt.format(type_.lower()))) for type_, fields_ in json.load(in_file).items()}

        return fields

    @staticmethod
    def __process(msg, mavlink, fields, init_fields, useful_types):
        msg_type = msg.get_type()
        to_remove = []
        for field, data in fields[msg_type].items():
            msg_data = getattr(msg, field)
            if data is None:
                fields[msg_type][field] = msg_data
            elif msg_data != data:
                if msg_type not in useful_types:
                    useful_types[msg_type] = []
                useful_types[msg_type].append(field)
                to_remove.append(field)
                if not fields[msg_type]:
                    mavlink.type_set.pop(msg_type)
                    init_fields.pop(msg_type)

        for field in to_remove:
            fields[msg_type].pop(field)

'''
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=None, type=str, help='Output filename for parsed telemetry data (defaults to tlog name with .csv extension)')
    parser.add_argument('--output_anomalies', default=None, type=str, help='Output filename for anomalies CSV')
    parser.add_argument('--output_comparisons', default=None, type=str, help='Output filename for comparisons CSV')
    parser.add_argument('-f', '--fields', default=None, type=str, help='Fields subset to parse with (json file)')
    parser.add_argument('-d', '--dialect', default='ardupilotmega', type=str, help='MAVLink dialect to parse with')
    parser.add_argument('-t', '--tlogs', required=True, nargs='*', help='Tlog filename(s) or path(s) to parse')
    parser.add_argument('-l', '--list', action='store_true', help='List useful (non-constant) fields')
    parser.add_argument('-q', '--quiet', action='store_true', help='Turn off printed output')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.list:
        fields = Telemetry.get_useful_fields(args.tlogs, args.output, args.fields, args.dialect, verbose)
        if verbose:
            print(json.dumps(fields, indent=4))
    else:
        fields = args.fields or Telemetry.DEFAULT_FIELDS
        for tlog in args.tlogs:
            telemetry = Telemetry(tlog, fields, args.dialect)
            telemetry.to_csv(
                output=args.output,
                anomalies_output=args.output_anomalies,
                comparisons_output=args.output_comparisons,
                verbose=verbose
            )
'''
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=None, type=str, help='Output filename for parsed telemetry data (defaults to tlog name with .csv extension)')
    parser.add_argument('--output_anomalies', default=None, type=str, help='Output filename for anomalies CSV')
    parser.add_argument('--output_comparisons', default=None, type=str, help='Output filename for comparisons CSV')
    parser.add_argument('-f', '--fields', default=None, type=str, help='Fields subset to parse with (json file)')
    parser.add_argument('-d', '--dialect', default='ardupilotmega', type=str, help='MAVLink dialect to parse with')
    parser.add_argument('-t', '--tlogs', required=True, nargs='*', help='Tlog filename(s) or path(s) to parse')
    parser.add_argument('-l', '--list', action='store_true', help='List useful (non-constant) fields')
    parser.add_argument('-q', '--quiet', action='store_true', help='Turn off printed output')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.list:
        fields = Telemetry.get_useful_fields(args.tlogs, args.output, args.fields, args.dialect, verbose)
        if verbose:
            print(json.dumps(fields, indent=4))
    else:
        fields = args.fields or Telemetry.DEFAULT_FIELDS

        # Create an instance of Telemetry (this will allow us to test comparison logic)
        telemetry = Telemetry(args.tlogs[0], fields, args.dialect)

        # --- BEGIN TEST CASE ---

        # Simulated dummy data to test the comparison logic
        data_list = [
            {
                'GPS2_RAW': {'vel': 3000},
                'GPS_RAW_INT': {'vel': 3050},
                'VFR_HUD': {'groundspeed': 30.5}
            },
            {
                'GPS2_RAW': {'vel': 3100},
                'GPS_RAW_INT': {'vel': 3120},
                'VFR_HUD': {'groundspeed': 31.2}
            }
        ]

        # Test the comparison logic
        comparison_results = telemetry.compare_similar_pairs(data_list)
        print(f"Dummy comparison results: {comparison_results}")

        # --- END TEST CASE ---

         #Comment out the following part if focusing only on the test case above
        for tlog in args.tlogs:
            telemetry = Telemetry(tlog, fields, args.dialect)
            telemetry.to_csv(
                output=args.output,
                anomalies_output=args.output_anomalies,
                comparisons_output=args.output_comparisons,
                verbose=verbose
            )
