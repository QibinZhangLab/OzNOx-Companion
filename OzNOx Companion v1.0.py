import copy
from datetime import datetime
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import exists
import pandas as pd
import re


# read_ozid_key() accepts a OzIDkey and returns an array of integers
# [n-#, neutral loss double-bonds, neutral loss hydroxyls]
# [] if invalid
def read_ozid_key(ozid_key):
    ozid_key = str(ozid_key)
    n_index = ozid_key.find('N')
    db_index = ozid_key.find('DB')
    oh_index = ozid_key.find('OH')
    if n_index < 0 or db_index < 0 or oh_index < 0:
        return []
    db_n_number = int(ozid_key[n_index + 1:db_index])
    neutral_loss_db = int(ozid_key[db_index + 2:oh_index])
    neutral_loss_oh = int(ozid_key[oh_index + 2:])
    return [db_n_number, neutral_loss_db, neutral_loss_oh]


# create_ozid_key() accepts an annotation string and returns an OzIDKey
def create_ozid_key(product_name, annotation_name):
    product_name_upper = product_name.upper()
    is_aldehyde = False
    is_criegee = False
    if product_name_upper.find('ALDEHYDE') >= 0:
        is_aldehyde = True
    if product_name_upper.find('CRIEGEE') >= 0:
        is_criegee = True
    multi_ozid = False
    plus_index = product_name.find(' + ')
    name_additions = []
    if plus_index > 0:
        multi_ozid = True
        product_name_parts = product_name.split(' + ')
        for t in range(len(product_name_parts)):
            if t != 0:
                name_additions += [product_name_parts[t]]
            else:
                name_additions += [product_name[len(annotation_name) + 1:]]
    else:
        name_additions = [product_name[len(annotation_name) + 1:]]
    n_numbers = []
    db_numbers = []
    oh_numbers = []
    for name_addition in name_additions:
        ozid_index = name_addition.find(' OzID ')
        n_number = name_addition[2:ozid_index]
        n_numbers += [n_number]
        with_index = name_addition.find(' with ')
        dbs_index = name_addition.find(' DBs, ')
        hydroxyls_index = name_addition.find(' hydroxyls ')
        db_number = name_addition[with_index + 6:dbs_index]
        db_numbers += [db_number]
        oh_number = name_addition[dbs_index + 6:hydroxyls_index]
        oh_numbers += [oh_number]
    ozid_key = '-1'
    if not multi_ozid:
        n_number = n_numbers[0]
        db_number = db_numbers[0]
        oh_number = oh_numbers[0]
        if is_aldehyde:
            ozid_key = 'AN' + str(n_number) + 'DB' + str(db_number) + 'OH' + str(oh_number)
        elif is_criegee:
            ozid_key = 'CN' + str(n_number) + 'DB' + str(db_number) + 'OH' + str(oh_number)
    else:
        ozid_keys = []
        w_16_index = product_name.find('wCriegee16')
        w_34_index = product_name.find('wCriegee34')
        w_48_index = product_name.find('wCriegee48')
        w_76_index = product_name.find('wCriegee76')
        for w in range(len(n_numbers)):
            n_number = n_numbers[w]
            db_number = db_numbers[w]
            oh_number = oh_numbers[w]
            if is_aldehyde:
                ozid_key = 'AN' + str(n_number) + 'DB' + str(db_number) + 'OH' + str(oh_number)
            elif is_criegee:
                ozid_key = 'CN' + str(n_number) + 'DB' + str(db_number) + 'OH' + str(oh_number)
            ozid_keys += [ozid_key]
        temp_string = ''
        for ozid_key in ozid_keys:
            if len(temp_string) > 1:
                temp_string += ' '
            temp_string += ozid_key
        if w_16_index > 0:
            temp_string += ' WC16'
        if w_34_index > 0:
            temp_string += ' WC34'
        if w_48_index > 0:
            temp_string += ' WC48'
        if w_76_index > 0:
            temp_string += ' WC76'
        ozid_key = temp_string

    return ozid_key


# is_digit() checks a single string character to see if it could be part of a number as 0-9
def is_digit(character):
    character = str(character)
    if (character == '0' or character == '1' or character == '2' or character == '3' or character == '4' or
            character == '5' or character == '6' or character == '7' or character == '8' or character == '9'):
        return True
    else:
        return False


# rank_int_arrays() ranks two integer arrays of equal length based on which array has the first greater integer
def rank_int_arrays(array_1, array_2):
    if len(array_1) != len(array_2):
        print('Error: arrays of unequal length compared.')
        return -1

    for i in range(len(array_1)):
        try:
            int_1 = int(array_1[i])
            int_2 = int(array_2[i])
        except Exception as e:
            print(e)
            print('Error: int-incompatible value fed to rank_int_arrays()')
            return -1
        if int_1 > int_2:
            return 1
        elif int_2 > int_1:
            return 2
    return 0


# time_stamp() returns a time stamp string in the format year_month_day_hour_minute_second
def time_stamp():
    now = str(datetime.now())
    now_halves = now.split(' ')
    now_date_half = now_halves[0]
    now_time_half = now_halves[1]
    now_date_pieces = now_date_half.split('-')
    now_time_pieces = now_time_half.split(':')
    now_seconds = now_time_pieces[2]
    now_seconds = now_seconds[:2]
    date_str = (now_date_pieces[0] + '_' + now_date_pieces[1] + '_' + now_date_pieces[2] + '_'
                + now_time_pieces[0] + '_' + now_time_pieces[1] + '_' + now_seconds)
    return date_str


# get_current_directory_files() returns an array of the current working directory's files sorted from oldest to newest
def get_current_directory_files():
    files_in_directory = [f for f in os.listdir() if os.path.isfile(f)]
    files_in_directory.sort(key=os.path.getctime)
    temp_array = []
    for file in files_in_directory:
        temp_array += [str(file)]
    files_in_directory = temp_array
    return files_in_directory


# convert_raw_txt_files_to_csv() converts MCConvert .txt spectra to .csv spectra
def convert_raw_txt_files_to_csv(file_name, ms_levels, minimum_signal, output_name):
    retention_times_array = []
    m_over_z_arrays = []
    intensities_arrays = []
    ms_levels_array = []
    precursors_array = []
    indexes = []

    past_spectrum = False
    ms_level = 0
    retention_time = 0
    m_over_z_array = []
    precursor_mz = -1
    good_ms_level = False
    mz_next = False
    intensity_next = False
    past_mz = False

    index = -1

    for line in open(file_name):

        spectrum_index = line.find('spectrum:')
        if spectrum_index >= 0:
            past_spectrum = True
            ms_level = 0
            retention_time = 0
            m_over_z_array = []
            precursor_mz = -1
            good_ms_level = False
            mz_next = False
            intensity_next = False
            past_mz = False

        ms_level_index = line.find('ms level,')
        if ms_level_index >= 0 and past_spectrum:
            ms_level = int(line[(ms_level_index + 10):(ms_level_index + 11)])
            for good_level in ms_levels:
                if ms_level == good_level:
                    good_ms_level = True
                    past_spectrum = False

        precursor_index = line.find('isolation window target m/z,')
        if precursor_index >= 0 and ms_level > 1 and good_ms_level:
            comma_mz_index = line.find(', m/z')
            precursor_mz = float(line[(precursor_index + 29):comma_mz_index])

        rt_index = line.find('scan start time,')
        if rt_index >= 0 and good_ms_level:
            comma_minute_index = line.find(', minute')
            retention_time = float(line[(rt_index + 17):comma_minute_index])

        mz_index = line.find('m/z array, m/z')
        if mz_index >= 0 and retention_time > 0:
            mz_next = True

        intensity_index = line.find('intensity array, ')
        if intensity_index >= 0 and retention_time > 0:
            intensity_next = True

        binary_index = line.find('binary: [')
        if binary_index >= 0 and (mz_next or intensity_next) and (not (mz_next and intensity_next)):
            end_bracket_index = line.find('] ')
            line = line[(end_bracket_index + 2): (len(line) - 2)]
            if mz_next:
                m_over_z_array = line
                mz_next = False
                past_mz = True
            elif intensity_next and past_mz:
                intensities_array = line

                retention_times_array += [retention_time]
                m_over_z_arrays += [m_over_z_array]
                intensities_arrays += [intensities_array]
                ms_levels_array += [ms_level]
                precursors_array += [precursor_mz]

                index += 1
                indexes += [index]

                past_spectrum = False
                ms_level = 0
                retention_time = 0
                m_over_z_array = []
                precursor_mz = 0
                good_ms_level = False
                mz_next = False
                intensity_next = False
                past_mz = False

    output_frame = pd.DataFrame(index=indexes, columns=['Scan RT', 'MS Level', 'MSn Precursor', 'Scan Spectrum'])
    for i in range(len(retention_times_array)):
        output_frame.loc[i]['Scan RT'] = retention_times_array[i]
        output_frame.loc[i]['MS Level'] = ms_levels_array[i]
        output_frame.loc[i]['MSn Precursor'] = precursors_array[i]
        mz_array = m_over_z_arrays[i]
        mz_array_split = mz_array.split(' ')
        intensity_array = intensities_arrays[i]
        intensity_array_split = intensity_array.split(' ')
        mz_intensity_array = ''
        for j in range(len(mz_array_split)):
            if (float(mz_array_split[j]) > 0) and (float(intensity_array_split[j]) > minimum_signal):
                mz_intensity_array += str(mz_array_split[j]) + ':' + str(intensity_array_split[j])
                if j < (len(mz_array_split) - 1):
                    mz_intensity_array += ' '
        output_frame.loc[i]['Scan Spectrum'] = mz_intensity_array
        temp_frame = output_frame.copy()
        output_frame = temp_frame.copy()
    output_frame.to_csv(output_name, index=False)
    print('Finished converting: ' + output_name)


# search_csv_data() searches .csv converted raw files for LC-MS features based on RT and m/z
# Return is an array of sums of signal intensity per mz_target in mz_targets array and RT of maximum intensity
# [[summed counts, RT of maximum count, RT:counts spectrum], ..., ...]
def search_csv_data(file_name, ms_level, rt_target, rt_tolerance, mz_targets, mz_tolerance, precursor_mz,
                    precursor_tolerance):
    try:
        raw_data_frame = pd.read_csv(file_name)
    except Exception as e:
        print(e)
        return -1
    raw_data_frame_index = raw_data_frame.index
    raw_data_frame_size = raw_data_frame_index.size

    precursor_mz_floor = precursor_mz - precursor_tolerance
    precursor_mz_ceiling = precursor_mz + precursor_tolerance

    ms_level = int(ms_level)
    rt_target = float(rt_target)
    rt_tolerance = float(rt_tolerance)
    temp_array = []
    for mz_target in mz_targets:
        mz_target = float(mz_target)
        temp_array += [mz_target]
    mz_targets = temp_array
    mz_tolerance = float(mz_tolerance)
    rt_floor = rt_target - rt_tolerance
    rt_ceiling = rt_target + rt_tolerance

    mz_floors = []
    mz_ceilings = []
    signal_sums = []
    max_intensities = []
    max_intensity_rts = []
    rt_counts_spectra = []
    for mz_target in mz_targets:
        mz_floor = mz_target - mz_tolerance
        mz_floors += [mz_floor]
        mz_ceiling = mz_target + mz_tolerance
        mz_ceilings += [mz_ceiling]
        signal_sums += [0.0]
        max_intensities += [0.0]
        max_intensity_rts += [-1]
        rt_counts_spectra += ['']

    # optimizing search start point
    below_floor = False
    matrix_index = int(raw_data_frame_size / 1.2)
    while not below_floor:
        matrix_row = raw_data_frame.loc[matrix_index]
        matrix_row_rt = float(matrix_row['Scan RT'])
        if matrix_index > 0 and matrix_row_rt > rt_floor:
            matrix_index = int(matrix_index / 1.2)
        else:
            below_floor = True

    # summing signal intensity with search optimization
    found_end = False
    while not found_end:
        if matrix_index == (raw_data_frame_size - 1):
            found_end = True
        else:
            matrix_row = raw_data_frame.loc[matrix_index]
            matrix_row_ms_level = int(matrix_row['MS Level'])
            if matrix_row_ms_level == ms_level:
                matrix_row_rt = float(matrix_row['Scan RT'])
                good_precursor = True
                if ms_level > 1:
                    row_precursor_mz = float(matrix_row['MSn Precursor'])
                    if precursor_mz_floor <= row_precursor_mz <= precursor_mz_ceiling:
                        good_precursor = True
                    else:
                        good_precursor = False
                if rt_floor <= matrix_row_rt <= rt_ceiling and good_precursor:
                    matrix_row_spectrum = str(matrix_row['Scan Spectrum'])
                    matrix_row_spectrum_elements = matrix_row_spectrum.split(' ')
                    num_spectrum_elements = len(matrix_row_spectrum_elements)
                    for i in range(len(mz_targets)):
                        below_floor = False
                        element_index = int(num_spectrum_elements / 1.2)
                        mz_floor = mz_floors[i]
                        mz_ceiling = mz_ceilings[i]
                        signal_sum = signal_sums[i]
                        new_sum = 0.0
                        max_intensity = max_intensities[i]
                        rt_counts_spectrum = rt_counts_spectra[i]
                        while not below_floor:
                            test_element = matrix_row_spectrum_elements[element_index]
                            test_element_split = test_element.split(':')
                            test_mz = float(test_element_split[0])
                            if element_index > 0 and test_mz > mz_floor:
                                element_index = int(element_index / 1.2)
                            else:
                                below_floor = True
                        mz_end = False
                        while not mz_end:
                            if element_index == (num_spectrum_elements - 1):
                                mz_end = True
                            else:
                                element = matrix_row_spectrum_elements[element_index]
                                element_split = element.split(':')
                                element_mz = float(element_split[0])
                                if mz_floor <= element_mz <= mz_ceiling:
                                    new_sum += float(element_split[1])
                                elif element_mz > mz_ceiling:
                                    mz_end = True
                            element_index += 1
                        signal_sums[i] = (signal_sum + new_sum)
                        if len(rt_counts_spectrum) <= 1:
                            rt_counts_spectrum += str(matrix_row_rt) + ':' + str(new_sum)
                        else:
                            rt_counts_spectrum += ' ' + str(matrix_row_rt) + ':' + str(new_sum)
                        rt_counts_spectra[i] = rt_counts_spectrum
                        if new_sum > max_intensity:
                            max_intensities[i] = new_sum
                            max_intensity_rts[i] = matrix_row_rt
                elif matrix_row_rt > rt_ceiling:
                    found_end = True
        matrix_index += 1

    output_array = []
    for i in range(len(signal_sums)):
        signal_sum_i = signal_sums[i]
        max_intensity_rt_i = max_intensity_rts[i]
        rt_counts_spectrum = rt_counts_spectra[i]
        output_array += [[signal_sum_i, max_intensity_rt_i, rt_counts_spectrum]]
    return output_array


# create_1_var_quad_equation() creates a quadratic equation with 1 independent variable, 1 dependent variable
def create_1_var_quad_equation(x, y):
    unique_xs = []
    for i in range(len(x)):
        x_i = x[i]
        new_x = True
        for unique_x in unique_xs:
            if unique_x == x_i:
                new_x = False
        if new_x:
            unique_xs += [x_i]
    if len(unique_xs) == 2:
        equation = np.polyfit(x, y, 1)
        slope = equation[0]
        intercept = equation[1]
        return [0, slope, intercept]
    elif len(unique_xs) < 2:
        return []
    try:
        equation = np.polyfit(x, y, 2)
        return equation
    except Exception as e:
        print(e)
        return []


# create_2_var_linear_equation() creates a linear equation with 2 independent variables, 1 dependent variable
def create_2_var_linear_equation(x1, x2, y):
    unique_x1s = []
    unique_x2s = []
    for i in range(len(x1)):
        x1_i = x1[i]
        x2_i = x2[i]
        new_x = True
        for unique_x1 in unique_x1s:
            if unique_x1 == x1_i:
                new_x = False
        if new_x:
            unique_x1s += [x1_i]
        new_x = True
        for unique_x2 in unique_x2s:
            if unique_x2 == x2_i:
                new_x = False
        if new_x:
            unique_x2s += [x2_i]
    if len(unique_x1s) < 2 or len(unique_x2s) < 2:
        return []

    indexes = []
    for i in range(len(x1)):
        indexes += [i]

    data = pd.DataFrame(columns=['x1', 'x2', 'y'], index=indexes)
    for i in indexes:
        data.loc[i]['x1'] = x1[i]
        data.loc[i]['x2'] = x2[i]
        data.loc[i]['y'] = y[i]

    x = data[['x1', 'x2']].values
    y = data['y'].values

    try:
        x = np.hstack([np.ones((x.shape[0], 1)), x])
    except Exception as e:
        print(e)
        return []

    x_train, x_test = x, x
    y_train, y_test = y, y

    x_train = np.array(x_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')

    try:
        x_train_t = x_train.T
        linear_equation_coefficients = np.linalg.inv(x_train_t @ x_train) @ x_train_t @ y_train
    except Exception as e:
        print(e)
        return []

    return linear_equation_coefficients


# Tail key object for use in the OzNOx scripts
class TailKey:

    tail_key_string = ''
    num_chains_expected = -1
    is_molecular_species = False
    is_summed_composition = False
    is_q_species = False
    is_p_species = False
    is_o_species = False
    is_d_species = False
    is_t_species = False
    is_m_species = False
    sn_positions_known = False
    num_chains = 0
    num_ethers = 0
    num_fatty_acyls = 0
    num_sphingoid_bases = 0
    raw_chains = []
    cleaned_chains = []
    simplified_chains = []
    num_carbons = 0
    num_doublebonds = 0
    num_hydroxyls = 0
    is_valid = True
    single_ozid_fragments = []
    tail_mass = 0
    ozid_aldehydes = []

    # calc_tail_mass() returns the mass of a single tail component
    # currently only intended for TailKey objects of a single tail mass
    def calc_tail_mass(self):
        if self.num_chains_expected != 1:
            self.tail_mass = -1
            return -1
        if self.num_carbons == 0:
            self.tail_mass = -1
            return -1
        if not self.is_valid:
            self.tail_mass = -1
            return -1
        if self.is_summed_composition:
            self.tail_mass = -1
            return -1
        mass_o16 = 15.994915
        mass_c12 = 12.0
        mass_h1 = 1.00784
        mass_n14 = 14.003074
        if self.is_q_species:
            tail_mass = (self.num_carbons * mass_c12 + (1 + 8 * (self.num_carbons / 5)) * mass_h1)
        elif self.is_p_species:
            tail_mass = ((1 + self.num_hydroxyls) * mass_o16 + self.num_carbons * mass_c12
                         + (2 * self.num_carbons + 1 - 2 * self.num_doublebonds) * mass_h1)
        elif self.is_o_species:
            tail_mass = ((1 + self.num_hydroxyls) * mass_o16 + self.num_carbons * mass_c12
                         + (2 * self.num_carbons + 1 - 2 * self.num_doublebonds) * mass_h1)
        elif self.is_d_species:
            tail_mass = self.num_carbons * mass_c12 + mass_n14 + 2 * mass_o16 + (2 * self.num_carbons + 2) * mass_h1
        elif self.is_t_species:
            tail_mass = self.num_carbons * mass_c12 + mass_n14 + 3 * mass_o16 + (2 * self.num_carbons + 2) * mass_h1
        else:
            tail_mass = ((2 + self.num_hydroxyls) * mass_o16 + self.num_carbons * mass_c12
                         + (2 * self.num_carbons + 1 - 2 - 2 * self.num_doublebonds) * mass_h1)
        self.tail_mass = tail_mass
        return tail_mass

    # create() determines the essential TailKey variables using a string in LipidSearch TailKey format and the number
    # of tail chains expected, e.g. 3 for the TG class or 2 for the Cer class
    def create(self, tail_key_string, num_chains_expected):

        tail_key_string = str(tail_key_string)
        self.tail_key_string = tail_key_string
        num_chains_expected = int(num_chains_expected)
        self.num_chains_expected = num_chains_expected
        self.is_molecular_species = False
        self.is_summed_composition = False
        self.is_q_species = False
        self.is_p_species = False
        self.is_o_species = False
        self.is_d_species = False
        self.is_t_species = False
        self.is_m_species = False
        self.num_chains = 0
        self.num_ethers = 0
        self.num_fatty_acyls = 0
        self.num_sphingoid_bases = 0
        split_string = '_|/'
        tail_key_string_split = re.split(split_string, tail_key_string)
        self.raw_chains = tail_key_string_split
        self.cleaned_chains = []
        self.simplified_chains = []
        self.num_carbons = 0
        self.num_doublebonds = 0
        self.num_hydroxyls = 0
        self.is_valid = True
        self.single_ozid_fragments = []
        self.tail_mass = 0
        self.ozid_aldehydes = []

        if len(tail_key_string_split) == num_chains_expected:
            self.is_molecular_species = True
        elif 0 < len(tail_key_string_split) < num_chains_expected:
            self.is_summed_composition = True
        else:
            print('Error: TailKey ' + tail_key_string + ' is invalid')
            self.is_valid = False
            return None
        self.num_chains = len(tail_key_string_split)
        for tail_key_part in tail_key_string_split:
            q_species = False
            p_species = False
            o_species = False
            m_species = False
            t_species = False
            d_species = False
            tail_key_shard_1 = ''
            tail_key_part_split = re.split(':', tail_key_part)
            if len(tail_key_part_split) == 1:
                tail_key_shard_0 = tail_key_part_split[0]
                if len(tail_key_shard_0) < 1:
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
                if tail_key_shard_0[0] == '(':
                    tail_key_shard_0 = tail_key_shard_0[1:]
                if tail_key_shard_0[0] == 'Q':
                    self.is_q_species = True
                    q_species = True
                    tail_key_shard_0 = tail_key_shard_0[1:]
                    if not is_digit(tail_key_shard_0[0]):
                        print('Error: TailKey ' + tail_key_string + ' is invalid')
                        self.is_valid = False
                        return None
                else:
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
            else:
                tail_key_shard_0 = tail_key_part_split[0]
                if len(tail_key_shard_0) < 1:
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
                tail_key_shard_1 = tail_key_part_split[1]
                if len(tail_key_shard_1) < 1:
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
                if tail_key_shard_0[0] == '(':
                    tail_key_shard_0 = tail_key_shard_0[1:]
                if len(tail_key_shard_0) > 2:
                    if tail_key_shard_0[0] == 'P' and tail_key_shard_0[1] == '-':
                        self.is_p_species = True
                        p_species = True
                        tail_key_shard_0 = tail_key_shard_0[2:]
                        self.num_ethers = self.num_ethers + 1
                    elif tail_key_shard_0[0] == 'O' and tail_key_shard_0[1] == '-':
                        self.is_o_species = True
                        o_species = True
                        tail_key_shard_0 = tail_key_shard_0[2:]
                        self.num_ethers = self.num_ethers + 1
                if len(tail_key_shard_0) > 1 and (not o_species) and (not p_species):
                    if tail_key_shard_0[0] == 'm':
                        self.num_hydroxyls += 1
                        m_species = True
                        self.is_m_species = True
                        tail_key_shard_0 = tail_key_shard_0[1:]
                    elif tail_key_shard_0[0] == 'd':
                        self.num_hydroxyls += 1
                        d_species = True
                        self.is_d_species = True
                        tail_key_shard_0 = tail_key_shard_0[1:]
                        self.num_sphingoid_bases = self.num_sphingoid_bases + 1
                    elif tail_key_shard_0[0] == 't':
                        self.num_hydroxyls += 2
                        t_species = True
                        self.is_t_species = True
                        tail_key_shard_0 = tail_key_shard_0[1:]
                        self.num_sphingoid_bases = self.num_sphingoid_bases + 1
                    else:
                        self.num_fatty_acyls = self.num_fatty_acyls + 1
                if len(tail_key_shard_0) == 0:
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
                if not is_digit(tail_key_shard_0[0]):
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
            chain_length_string = ''
            found_end = False
            for char in tail_key_shard_0:
                if is_digit(char) and not found_end:
                    chain_length_string += char
                else:
                    found_end = True
            try:
                chain_length = int(chain_length_string)
            except Exception as e:
                print(e)
                print('Error: TailKey ' + tail_key_string + ' is invalid')
                self.is_valid = False
                return None
            doublebond_string = ''
            num_dbs = 0
            if len(tail_key_part_split) != 1:
                found_end = False
                for char in tail_key_shard_1:
                    if is_digit(char) and not found_end:
                        doublebond_string += char
                    else:
                        found_end = True
                try:
                    num_dbs = int(doublebond_string)
                except Exception as e:
                    print(e)
                    print('Error: TailKey ' + tail_key_string + ' is invalid')
                    self.is_valid = False
                    return None
            if q_species:
                self.num_carbons += (chain_length * 5)
                self.num_doublebonds += chain_length
                cleaned_chain = 'Q' + str(chain_length)
                self.cleaned_chains += [cleaned_chain]
                simplified_chain = str(chain_length * 5) + ':' + str(chain_length)
                self.simplified_chains += [simplified_chain]
            else:
                self.num_carbons += chain_length
                if p_species:
                    num_dbs += 1
                self.num_doublebonds += num_dbs
                if p_species:
                    cleaned_chain = 'P-'
                elif o_species:
                    cleaned_chain = 'O-'
                elif m_species:
                    cleaned_chain = 'm'
                elif d_species:
                    cleaned_chain = 'd'
                elif t_species:
                    cleaned_chain = 't'
                else:
                    cleaned_chain = ''
                if p_species:
                    num_dbs -= 1
                cleaned_chain = cleaned_chain + str(chain_length) + ':' + str(num_dbs)
                self.cleaned_chains += [cleaned_chain]
                if p_species:
                    num_dbs += 1
                simplified_chain = str(chain_length) + ':' + str(num_dbs)
                self.simplified_chains += [simplified_chain]

        underscore_index = tail_key_string.find('_')
        if self.is_summed_composition:
            self.sn_positions_known = False
        elif underscore_index >= 0:
            self.sn_positions_known = False
        else:
            self.sn_positions_known = True

    # generate_single_ozid_products() returns an array of arrays containing paired OzID ion names and mz values.
    # The list of OzID generates is a shorter list due to only producing ions with one ozone induced dissociation.
    # A much longer list would be created if all multiple dissociation possibilities were explored.
    def generate_single_ozid_products(self, precursor_name, precursor_mz, precursor_z, has_criegee, has_methanol,
                                      has_ipa):
        return_array = []
        if self.num_chains_expected <= 0 or self.num_chains_expected >= 6:
            print('Error: bad num_chains_expected given for TailKey ' + self.tail_key_string)
            return return_array
        if has_methanol and has_ipa:
            print('Error: OzID attempted with both methanol and IPA.')
            return return_array
        ozid_aldehyde_names = []
        mass_o16 = 15.994915
        mass_c12 = 12.0
        mass_h1 = 1.00784
        precursor_name = str(precursor_name)
        precursor_mz = float(precursor_mz)
        precursor_z = int(precursor_z)
        precursor_mass = precursor_mz * precursor_z
        for precursor_chain in self.cleaned_chains:
            chain_key = TailKey()
            chain_key.create(precursor_chain, 1)
            p_species = chain_key.is_p_species
            o_species = chain_key.is_o_species
            d_species = chain_key.is_d_species
            q_species = chain_key.is_q_species
            t_species = chain_key.is_t_species
            chain_length = chain_key.num_carbons
            banned_n_no = []
            # banned_n_no += [(chain_length - 5)]
            chain_doublebonds = chain_key.num_doublebonds
            chain_hydroxyls = chain_key.num_hydroxyls
            if not q_species and chain_doublebonds > 0:
                if d_species or t_species:
                    max_chain_position = chain_length - 4
                elif p_species:
                    max_chain_position = chain_length - 4
                elif o_species:
                    max_chain_position = chain_length - 3
                else:
                    max_chain_position = chain_length - 2
                if not self.is_molecular_species:
                    max_chain_position = max_chain_position - 2 * (self.num_chains_expected - self.num_chains)
                current_chain_position = 0
                while current_chain_position < max_chain_position:
                    current_chain_position += 1
                    legal_pos = True
                    for banned_n in banned_n_no:
                        if banned_n == current_chain_position:
                            legal_pos = False
                    if legal_pos:
                        num_dbs_behind = -2
                        total_dbs = 0
                        max_possible_dbs_behind = float(current_chain_position) / 2.0
                        remainder = max_possible_dbs_behind % 1.0
                        if remainder == 0.0:
                            max_possible_dbs_behind -= 1.0
                        max_possible_dbs_behind = int(max_possible_dbs_behind)
                        if max_possible_dbs_behind > (chain_doublebonds - 1):
                            max_possible_dbs_behind = (chain_doublebonds - 1)
                        max_possible_dbs_ahead = int((max_chain_position - current_chain_position) / 2)
                        if max_possible_dbs_ahead > (chain_doublebonds - 1):
                            max_possible_dbs_ahead = (chain_doublebonds - 1)
                        max_possible_oh_behind = chain_hydroxyls
                        if d_species:
                            max_possible_oh_behind -= 1
                        if max_possible_oh_behind > current_chain_position:
                            max_possible_oh_behind = current_chain_position
                        if max_possible_oh_behind < 0:
                            max_possible_oh_behind = 0
                        while total_dbs < chain_doublebonds and num_dbs_behind < max_possible_dbs_behind:
                            num_dbs_behind += 1
                            if num_dbs_behind == -1:
                                num_dbs_behind = 0
                            if total_dbs == 0:
                                total_dbs = 1
                            if num_dbs_behind > 0:
                                total_dbs += 1
                            num_dbs_ahead = chain_doublebonds - 1 - num_dbs_behind
                            if num_dbs_ahead <= max_possible_dbs_ahead:
                                num_oh_behind = -1
                                while num_oh_behind <= max_possible_oh_behind:
                                    num_oh_behind += 1
                                    if num_oh_behind <= max_possible_oh_behind:
                                        aldehyde_mass = ((precursor_mass - current_chain_position * mass_c12
                                                          - (2 * current_chain_position - 2 * num_dbs_behind) * mass_h1)
                                                         + (1 - num_oh_behind) * mass_o16)
                                        if has_methanol:
                                            criegee_mass = aldehyde_mass + 2 * mass_o16 + mass_c12 + 4 * mass_h1
                                        elif has_ipa:
                                            criegee_mass = aldehyde_mass + 2 * mass_o16 + 3 * mass_c12 + 8 * mass_h1
                                        else:
                                            criegee_mass = aldehyde_mass + mass_o16
                                        aldehyde_mz = aldehyde_mass * precursor_z
                                        aldehyde_mz = round(aldehyde_mz, 6)
                                        criegee_mz = criegee_mass * precursor_z
                                        criegee_mz = round(criegee_mz, 6)
                                        aldehyde_name = (precursor_name + ' n-' + str(current_chain_position)
                                                         + ' OzID aldehyde with ' + str(num_dbs_behind)
                                                         + ' DBs, ' + str(num_oh_behind) + ' hydroxyls in neutral loss')
                                        criegee_name = (precursor_name + ' n-' + str(current_chain_position)
                                                        + ' OzID Criegee with ' + str(num_dbs_behind)
                                                        + ' DBs, ' + str(num_oh_behind) + ' hydroxyls in neutral loss')
                                        novel_fragments = True
                                        for ozid_aldehyde_name in ozid_aldehyde_names:
                                            if ozid_aldehyde_name == aldehyde_name:
                                                novel_fragments = False
                                        if novel_fragments:
                                            ozid_aldehyde_names += [aldehyde_name]
                                            return_array += [[aldehyde_name, aldehyde_mz]]
                                            if has_criegee:
                                                return_array += [[criegee_name, criegee_mz]]
        self.single_ozid_fragments = return_array
        return return_array

    # generate_multi_ozid_aldehydes() returns an array of arrays containing paired OzID ion names and mz values.
    # The list of OzID generates is a longer list due to finding products resulting from more than one OzID reaction
    def generate_multi_ozid_aldehydes(self, precursor_name, precursor_mz, precursor_z, ozid_level):

        two_chain_permutations = [[1, 2]]
        three_chain_permutations = [[1, 2], [1, 3], [2, 3], [1, 2, 3]]
        four_chain_permutations = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4],
                                   [2, 3, 4], [1, 2, 3, 4]]
        ozid_aldehyde_names = []
        if self.num_chains_expected <= 0 or self.num_chains_expected >= 6:
            print('Error: bad num_chains_expected given for TailKey ' + self.tail_key_string)
            return None
        mass_o16 = 15.994915
        mass_c12 = 12.0
        mass_h1 = 1.00784
        precursor_name = str(precursor_name)
        precursor_mz = float(precursor_mz)
        precursor_z = int(precursor_z)
        precursor_mass = precursor_mz * precursor_z
        self.generate_single_ozid_products(precursor_name, precursor_mz, precursor_z, False,
                                           False, False)
        return_array = self.single_ozid_fragments
        temp_array = []
        for product_pair in return_array:
            product_name = product_pair[0]
            product_name_upper = product_name.upper()
            aldehyde_index = product_name_upper.find('ALDEHYDE')
            if aldehyde_index >= 0:
                temp_array += [[product_name, product_pair[1]]]
                ozid_aldehyde_names += [product_name]
        return_array = temp_array
        num_chains_with_dbs = 0
        chains_with_dbs = []
        chain_permutations = []
        for precursor_chain in self.cleaned_chains:
            chain_key = TailKey()
            chain_key.create(precursor_chain, 1)
            num_dbs = chain_key.num_doublebonds
            q_species = chain_key.is_q_species
            if not q_species and num_dbs > 0:
                num_chains_with_dbs += 1
                chains_with_dbs += [precursor_chain]
        if num_chains_with_dbs == 0:
            print('Error: OzID calculation attempted with incompatible Tail Key.')
            return []
        elif num_chains_with_dbs == 1:
            self.ozid_aldehydes = return_array
            return None
        elif num_chains_with_dbs == 2:
            chain_permutations = two_chain_permutations
        elif num_chains_with_dbs == 3:
            chain_permutations = three_chain_permutations
        elif num_chains_with_dbs == 4:
            chain_permutations = four_chain_permutations
        for chain_permutation in chain_permutations:
            if len(chain_permutation) <= ozid_level:
                permutated_chains = []
                for index in chain_permutation:
                    permutated_chains += [chains_with_dbs[(index - 1)]]
                mass_losses_arrays = []
                for precursor_chain in permutated_chains:
                    mass_losses_array = []
                    chain_key = TailKey()
                    chain_key.create(precursor_chain, 1)
                    p_species = chain_key.is_p_species
                    o_species = chain_key.is_o_species
                    d_species = chain_key.is_d_species
                    t_species = chain_key.is_t_species
                    chain_length = chain_key.num_carbons
                    chain_doublebonds = chain_key.num_doublebonds
                    chain_hydroxyls = chain_key.num_hydroxyls
                    if d_species or t_species:
                        max_chain_position = chain_length - 4
                    elif p_species:
                        max_chain_position = chain_length - 4
                    elif o_species:
                        max_chain_position = chain_length - 3
                    else:
                        max_chain_position = chain_length - 2
                    current_chain_position = 0
                    while current_chain_position < max_chain_position:
                        current_chain_position += 1
                        num_dbs_behind = -2
                        total_dbs = 0
                        max_possible_dbs_behind = float(current_chain_position) / 2.0
                        remainder = max_possible_dbs_behind % 1.0
                        if remainder == 0.0:
                            max_possible_dbs_behind -= 1.0
                        max_possible_dbs_behind = int(max_possible_dbs_behind)
                        if max_possible_dbs_behind > (chain_doublebonds - 1):
                            max_possible_dbs_behind = (chain_doublebonds - 1)
                        max_possible_dbs_ahead = int((max_chain_position - current_chain_position) / 2)
                        if max_possible_dbs_ahead > (chain_doublebonds - 1):
                            max_possible_dbs_ahead = (chain_doublebonds - 1)
                        max_possible_oh_behind = chain_hydroxyls
                        if d_species:
                            max_possible_oh_behind -= 1
                        if max_possible_oh_behind > current_chain_position:
                            max_possible_oh_behind = current_chain_position
                        if max_possible_oh_behind < 0:
                            max_possible_oh_behind = 0
                        while total_dbs < chain_doublebonds and num_dbs_behind < max_possible_dbs_behind:
                            num_dbs_behind += 1
                            if num_dbs_behind == -1:
                                num_dbs_behind = 0
                            if total_dbs == 0:
                                total_dbs = 1
                            if num_dbs_behind > 0:
                                total_dbs += 1
                            num_dbs_ahead = chain_doublebonds - 1 - num_dbs_behind
                            if num_dbs_ahead <= max_possible_dbs_ahead:
                                num_oh_behind = -1
                                while num_oh_behind <= max_possible_oh_behind:
                                    num_oh_behind += 1
                                    if num_oh_behind <= max_possible_oh_behind:
                                        mass_loss = (current_chain_position * mass_c12
                                                     + (2 * current_chain_position - 2 * num_dbs_behind) * mass_h1
                                                     - (1 - num_oh_behind) * mass_o16)
                                        aldehyde_name = ('n-' + str(current_chain_position)
                                                         + ' OzID aldehyde with ' + str(num_dbs_behind)
                                                         + ' DBs, ' + str(num_oh_behind) + ' hydroxyls in neutral loss')
                                        mass_loss_code = [current_chain_position, num_dbs_behind, num_oh_behind]
                                        mass_losses_array += [[aldehyde_name, mass_loss, mass_loss_code]]
                    mass_losses_arrays += [[mass_losses_array]]
                if len(mass_losses_arrays) == 2:
                    mass_losses_1 = mass_losses_arrays[0]
                    mass_losses_2 = mass_losses_arrays[1]
                    mass_losses_1 = mass_losses_1[0]
                    mass_losses_2 = mass_losses_2[0]
                    for mass_loss_1 in mass_losses_1:
                        mass_loss_1_name = mass_loss_1[0]
                        mass_loss_1_mass = mass_loss_1[1]
                        mass_loss_1_code = mass_loss_1[2]
                        for mass_loss_2 in mass_losses_2:
                            mass_loss_2_name = mass_loss_2[0]
                            mass_loss_2_mass = mass_loss_2[1]
                            mass_loss_2_code = mass_loss_2[2]
                            product_mass = precursor_mass - mass_loss_1_mass - mass_loss_2_mass
                            product_mz = product_mass / precursor_z
                            rank_result = rank_int_arrays(mass_loss_1_code, mass_loss_2_code)
                            if rank_result == 2:
                                product_name = precursor_name + ' ' + mass_loss_2_name + ' + ' + mass_loss_1_name
                            else:
                                product_name = precursor_name + ' ' + mass_loss_1_name + ' + ' + mass_loss_2_name
                            novel_product = True
                            for old_product in ozid_aldehyde_names:
                                if product_name == old_product:
                                    novel_product = False
                            if novel_product:
                                ozid_aldehyde_names += [product_name]
                                return_array += [[product_name, product_mz]]
                elif len(mass_losses_arrays) == 3:
                    mass_losses_1 = mass_losses_arrays[0]
                    mass_losses_2 = mass_losses_arrays[1]
                    mass_losses_3 = mass_losses_arrays[2]
                    mass_losses_1 = mass_losses_1[0]
                    mass_losses_2 = mass_losses_2[0]
                    mass_losses_3 = mass_losses_3[0]
                    for mass_loss_1 in mass_losses_1:
                        mass_loss_1_name = mass_loss_1[0]
                        mass_loss_1_mass = mass_loss_1[1]
                        mass_loss_1_code = mass_loss_1[2]
                        for mass_loss_2 in mass_losses_2:
                            mass_loss_2_name = mass_loss_2[0]
                            mass_loss_2_mass = mass_loss_2[1]
                            mass_loss_2_code = mass_loss_2[2]
                            for mass_loss_3 in mass_losses_3:
                                mass_loss_3_name = mass_loss_3[0]
                                mass_loss_3_mass = mass_loss_3[1]
                                mass_loss_3_code = mass_loss_3[2]
                                product_mass = precursor_mass - mass_loss_1_mass - mass_loss_2_mass - mass_loss_3_mass
                                product_mz = product_mass / precursor_z
                                rank_order_1_3_2 = False
                                rank_order_2_1_3 = False
                                rank_order_2_3_1 = False
                                rank_order_3_2_1 = False
                                rank_order_3_1_2 = False
                                rank_result_1_2 = rank_int_arrays(mass_loss_1_code, mass_loss_2_code)
                                rank_result_1_3 = rank_int_arrays(mass_loss_1_code, mass_loss_3_code)
                                rank_result_2_3 = rank_int_arrays(mass_loss_2_code, mass_loss_3_code)
                                if rank_result_1_2 == 1 and rank_result_1_3 == 1:
                                    if rank_result_2_3 == 2:
                                        rank_order_1_3_2 = True
                                elif rank_result_1_2 == 2 and rank_result_2_3 == 1:
                                    if rank_result_1_3 == 2:
                                        rank_order_2_3_1 = True
                                    else:
                                        rank_order_2_1_3 = True
                                elif rank_result_1_3 == 2 and rank_result_2_3 == 2:
                                    if rank_result_1_2 == 2:
                                        rank_order_3_2_1 = True
                                    else:
                                        rank_order_3_1_2 = True
                                else:
                                    if rank_result_2_3 == 0 and rank_result_1_3 == 2:
                                        rank_order_2_3_1 = True
                                    elif rank_result_1_3 == 0 and rank_result_2_3 == 2:
                                        rank_order_1_3_2 = True

                                if rank_order_1_3_2:
                                    product_name = (precursor_name + ' ' + mass_loss_1_name + ' + ' + mass_loss_3_name
                                                    + ' + ' + mass_loss_2_name)
                                elif rank_order_2_3_1:
                                    product_name = (precursor_name + ' ' + mass_loss_2_name + ' + ' + mass_loss_3_name
                                                    + ' + ' + mass_loss_1_name)
                                elif rank_order_2_1_3:
                                    product_name = (precursor_name + ' ' + mass_loss_2_name + ' + ' + mass_loss_1_name
                                                    + ' + ' + mass_loss_3_name)
                                elif rank_order_3_2_1:
                                    product_name = (precursor_name + ' ' + mass_loss_3_name + ' + ' + mass_loss_2_name
                                                    + ' + ' + mass_loss_1_name)
                                elif rank_order_3_1_2:
                                    product_name = (precursor_name + ' ' + mass_loss_3_name + ' + ' + mass_loss_1_name
                                                    + ' + ' + mass_loss_2_name)
                                else:
                                    product_name = (precursor_name + ' ' + mass_loss_1_name + ' + ' + mass_loss_2_name
                                                    + ' + ' + mass_loss_3_name)
                                novel_product = True
                                for old_product in ozid_aldehyde_names:
                                    if product_name == old_product:
                                        novel_product = False
                                if novel_product:
                                    ozid_aldehyde_names += [product_name]
                                    return_array += [[product_name, product_mz]]
                elif len(mass_losses_arrays) == 4:
                    mass_losses_1 = mass_losses_arrays[0]
                    mass_losses_2 = mass_losses_arrays[1]
                    mass_losses_3 = mass_losses_arrays[2]
                    mass_losses_4 = mass_losses_arrays[3]
                    mass_losses_1 = mass_losses_1[0]
                    mass_losses_2 = mass_losses_2[0]
                    mass_losses_3 = mass_losses_3[0]
                    mass_losses_4 = mass_losses_4[0]
                    for mass_loss_1 in mass_losses_1:
                        mass_loss_1_name = mass_loss_1[0]
                        mass_loss_1_mass = mass_loss_1[1]
                        mass_loss_1_code = mass_loss_1[2]
                        for mass_loss_2 in mass_losses_2:
                            mass_loss_2_name = mass_loss_2[0]
                            mass_loss_2_mass = mass_loss_2[1]
                            mass_loss_2_code = mass_loss_2[2]
                            for mass_loss_3 in mass_losses_3:
                                mass_loss_3_name = mass_loss_3[0]
                                mass_loss_3_mass = mass_loss_3[1]
                                mass_loss_3_code = mass_loss_3[2]
                                for mass_loss_4 in mass_losses_4:
                                    mass_loss_4_name = mass_loss_4[0]
                                    mass_loss_4_mass = mass_loss_4[1]
                                    mass_loss_4_code = mass_loss_4[2]
                                    product_mass = (precursor_mass - mass_loss_1_mass - mass_loss_2_mass
                                                    - mass_loss_3_mass - mass_loss_4_mass)
                                    product_mz = product_mass / precursor_z

                                    rank_order_1_2_4_3 = False
                                    rank_order_1_3_2_4 = False
                                    rank_order_1_3_4_2 = False
                                    rank_order_1_4_2_3 = False
                                    rank_order_1_4_3_2 = False
                                    rank_order_2_1_3_4 = False
                                    rank_order_2_1_4_3 = False
                                    rank_order_2_3_1_4 = False
                                    rank_order_2_3_4_1 = False
                                    rank_order_2_4_1_3 = False
                                    rank_order_2_4_3_1 = False
                                    rank_order_3_2_1_4 = False
                                    rank_order_3_2_4_1 = False
                                    rank_order_3_1_2_4 = False
                                    rank_order_3_1_4_2 = False
                                    rank_order_3_4_2_1 = False
                                    rank_order_3_4_1_2 = False
                                    rank_order_4_2_3_1 = False
                                    rank_order_4_2_1_3 = False
                                    rank_order_4_3_2_1 = False
                                    rank_order_4_3_1_2 = False
                                    rank_order_4_1_2_3 = False
                                    rank_order_4_1_3_2 = False

                                    rank_result_1_2 = rank_int_arrays(mass_loss_1_code, mass_loss_2_code)
                                    rank_result_1_3 = rank_int_arrays(mass_loss_1_code, mass_loss_3_code)
                                    rank_result_1_4 = rank_int_arrays(mass_loss_1_code, mass_loss_4_code)
                                    rank_result_2_3 = rank_int_arrays(mass_loss_2_code, mass_loss_3_code)
                                    rank_result_2_4 = rank_int_arrays(mass_loss_2_code, mass_loss_4_code)
                                    rank_result_3_4 = rank_int_arrays(mass_loss_3_code, mass_loss_4_code)

                                    if rank_result_1_2 == 1 and rank_result_1_3 == 1 and rank_result_1_4 == 1:
                                        if rank_result_2_3 == 1 and rank_result_2_4 == 1:
                                            if rank_result_3_4 == 2:
                                                rank_order_1_2_4_3 = True
                                        elif rank_result_2_3 == 2 and rank_result_3_4 == 1:
                                            if rank_result_2_4 == 2:
                                                rank_order_1_3_4_2 = True
                                            else:
                                                rank_order_1_3_2_4 = True
                                        elif rank_result_2_4 == 2 and rank_result_3_4 == 2:
                                            if rank_result_2_3 == 2:
                                                rank_order_1_4_3_2 = True
                                            else:
                                                rank_order_1_4_2_3 = True
                                        else:
                                            if rank_result_2_3 == 0:
                                                if rank_result_2_4 == 2:
                                                    rank_order_1_4_2_3 = True
                                            elif rank_result_2_4 == 0:
                                                if rank_result_2_3 == 2:
                                                    rank_order_1_3_2_4 = True
                                                else:
                                                    rank_order_1_2_4_3 = True
                                            elif rank_result_3_4 == 0:
                                                if rank_result_2_3 == 2:
                                                    rank_order_1_3_4_2 = True
                                    elif rank_result_1_2 == 2 and rank_result_2_3 == 1 and rank_result_2_4 == 1:
                                        if rank_result_1_3 == 1 and rank_result_1_4 == 1:
                                            if rank_result_3_4 == 2:
                                                rank_order_2_1_4_3 = True
                                            else:
                                                rank_order_2_1_3_4 = True
                                        elif rank_result_1_3 == 2 and rank_result_3_4 == 1:
                                            if rank_result_1_4 == 2:
                                                rank_order_2_3_4_1 = True
                                            else:
                                                rank_order_2_3_1_4 = True
                                        elif rank_result_1_4 == 2 and rank_result_3_4 == 2:
                                            if rank_result_1_3 == 2:
                                                rank_order_2_4_3_1 = True
                                            else:
                                                rank_order_2_4_1_3 = True
                                        else:
                                            if rank_result_1_3 == 0:
                                                if rank_result_1_4 == 2:
                                                    rank_order_2_4_1_3 = True
                                                else:
                                                    rank_order_2_1_3_4 = True
                                            elif rank_result_1_4 == 0:
                                                if rank_result_1_3 == 2:
                                                    rank_order_2_3_1_4 = True
                                                else:
                                                    rank_order_2_1_4_3 = True
                                            elif rank_result_3_4 == 0:
                                                if rank_result_1_3 == 1:
                                                    rank_order_2_1_3_4 = True
                                                else:
                                                    rank_order_2_3_4_1 = True
                                    elif rank_result_1_3 == 2 and rank_result_2_3 == 2 and rank_result_3_4 == 1:
                                        if rank_result_1_2 == 1 and rank_result_1_4 == 1:
                                            if rank_result_2_4 == 2:
                                                rank_order_3_1_4_2 = True
                                            else:
                                                rank_order_3_1_2_4 = True
                                        elif rank_result_1_2 == 2 and rank_result_2_4 == 1:
                                            if rank_result_1_4 == 2:
                                                rank_order_3_2_4_1 = True
                                            else:
                                                rank_order_3_2_1_4 = True
                                        elif rank_result_1_4 == 2 and rank_result_2_4 == 2:
                                            if rank_result_1_2 == 2:
                                                rank_order_3_4_2_1 = True
                                            else:
                                                rank_order_3_4_1_2 = True
                                        else:
                                            if rank_result_1_2 == 0:
                                                if rank_result_1_4 == 2:
                                                    rank_order_3_4_1_2 = True
                                                else:
                                                    rank_order_3_1_2_4 = True
                                            elif rank_result_1_4 == 0:
                                                if rank_result_1_2 == 2:
                                                    rank_order_3_2_1_4 = True
                                                else:
                                                    rank_order_3_1_4_2 = True
                                            elif rank_result_2_4 == 0:
                                                if rank_result_1_2 == 1:
                                                    rank_order_3_1_2_4 = True
                                                else:
                                                    rank_order_3_2_4_1 = True
                                    elif rank_result_1_4 == 2 and rank_result_2_4 == 2 and rank_result_3_4 == 2:
                                        if rank_result_1_2 == 1 and rank_result_1_3 == 1:
                                            if rank_result_2_3 == 2:
                                                rank_order_4_1_3_2 = True
                                            else:
                                                rank_order_4_1_2_3 = True
                                        elif rank_result_1_2 == 2 and rank_result_2_3 == 1:
                                            if rank_result_1_3 == 2:
                                                rank_order_4_2_3_1 = True
                                            else:
                                                rank_order_4_2_1_3 = True
                                        elif rank_result_1_3 == 2 and rank_result_2_3 == 2:
                                            if rank_result_1_2 == 2:
                                                rank_order_4_3_2_1 = True
                                            else:
                                                rank_order_4_3_1_2 = True
                                        else:
                                            if rank_result_1_2 == 0:
                                                if rank_result_1_3 == 2:
                                                    rank_order_4_3_1_2 = True
                                                else:
                                                    rank_order_4_1_2_3 = True
                                            elif rank_result_1_3 == 0:
                                                if rank_result_1_2 == 2:
                                                    rank_order_4_2_1_3 = True
                                                else:
                                                    rank_order_4_1_3_2 = True
                                            elif rank_result_2_3 == 0:
                                                if rank_result_1_2 == 1:
                                                    rank_order_4_1_2_3 = True
                                                else:
                                                    rank_order_4_2_3_1 = True
                                    else:
                                        if rank_result_1_2 == 0 and not (rank_result_1_3 == 2 or rank_result_1_4 == 2):
                                            if rank_result_3_4 == 2:
                                                rank_order_1_2_4_3 = True
                                        elif (rank_result_1_3 == 0
                                              and not (rank_result_1_2 == 2 or rank_result_1_4 == 2)):
                                            if rank_result_2_4 == 2:
                                                rank_order_1_3_4_2 = True
                                            else:
                                                rank_order_1_3_2_4 = True
                                        elif (rank_result_1_4 == 0
                                              and not (rank_result_1_2 == 2 or rank_result_1_3 == 2)):
                                            if rank_result_2_3 == 2:
                                                rank_order_1_4_3_2 = True
                                            else:
                                                rank_order_1_4_2_3 = True
                                        elif (rank_result_2_3 == 0
                                              and not (rank_result_1_2 == 1 or rank_result_2_4 == 2)):
                                            if rank_result_1_4 == 2:
                                                rank_order_2_3_4_1 = True
                                            else:
                                                rank_order_2_3_1_4 = True
                                        elif (rank_result_2_4 == 0
                                              and not (rank_result_1_2 == 1 or rank_result_3_4 == 1)):
                                            if rank_result_1_3 == 2:
                                                rank_order_2_4_3_1 = True
                                            else:
                                                rank_order_2_4_1_3 = True
                                        elif (rank_result_3_4 == 0
                                              and not (rank_result_1_3 == 1 or rank_result_2_3 == 1)):
                                            if rank_result_1_2 == 2:
                                                rank_order_3_4_2_1 = True
                                            else:
                                                rank_order_3_4_1_2 = True

                                    if rank_order_1_2_4_3:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_1_3_2_4:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_4_name)
                                    elif rank_order_1_3_4_2:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_2_name)
                                    elif rank_order_1_4_2_3:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_1_4_3_2:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_2_name)
                                    elif rank_order_2_1_3_4:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_4_name)
                                    elif rank_order_2_1_4_3:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_2_3_1_4:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_4_name)
                                    elif rank_order_2_3_4_1:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_2_4_1_3:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_2_4_3_1:
                                        product_name = (precursor_name + ' ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_3_2_1_4:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_4_name)
                                    elif rank_order_3_2_4_1:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_3_1_2_4:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_4_name)
                                    elif rank_order_3_1_4_2:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_4_name + ' + ' + mass_loss_2_name)
                                    elif rank_order_3_4_2_1:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_3_4_1_2:
                                        product_name = (precursor_name + ' ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_2_name)
                                    elif rank_order_4_2_3_1:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_4_2_1_3:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_4_3_2_1:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_1_name)
                                    elif rank_order_4_3_1_2:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_3_name
                                                        + ' + ' + mass_loss_1_name + ' + ' + mass_loss_2_name)
                                    elif rank_order_4_1_2_3:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_2_name + ' + ' + mass_loss_3_name)
                                    elif rank_order_4_1_3_2:
                                        product_name = (precursor_name + ' ' + mass_loss_4_name
                                                        + ' + ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_2_name)
                                    else:
                                        product_name = (precursor_name + ' ' + mass_loss_1_name
                                                        + ' + ' + mass_loss_2_name
                                                        + ' + ' + mass_loss_3_name + ' + ' + mass_loss_4_name)
                                    novel_product = True
                                    for old_product in ozid_aldehyde_names:
                                        if product_name == old_product:
                                            novel_product = False
                                    if novel_product:
                                        ozid_aldehyde_names += [product_name]
                                        return_array += [[product_name, product_mz]]
        self.ozid_aldehydes = return_array


# OzNOx Script 1 inputs the project folder directory and core definitions files
def oznox_script_1():

    print('')
    print('Beginning Script 1 ...')

    # Section below establishes project directory (folder) and checks the parameter files
    project_directory = None
    project_directory_files = None
    class_definitions_file = None
    class_definitions_frame = None
    class_definitions_frame_size = None
    class_definitions_classes = None
    row_num_tail_chains = None
    row_num_fatty_acyls = None
    row_num_sphingoid_bases = None
    row_oznox_scheme_1_bool = None
    row_oznox_scheme_1_mass = None
    row_oznox_scheme_2_bool = None
    row_oznox_scheme_2_mass = None
    lcms_annotations_file = None
    lcms_annotations_frame = None
    lcms_annotations_frame_size = None
    row_parent_id = None
    row_annotation_name = None
    row_reject = None
    row_class_key = None
    row_tail_key = None
    row_tail_key_object = None
    row_mz = None
    row_rt = None
    row_z = None
    class_definitions_required_columns = ['Class Key', 'Num Tail Chains', 'Num Fatty Acyls', 'Num Sphingoid Bases',
                                          'OzNOx MS1 m/z Shift', 'OzNOx MS2 Collision Energy',
                                          'OzNOx MS2 Product Ion Scheme 1', 'Scheme 1 Additional Mass Loss',
                                          'OzNOx MS2 Product Ion Scheme 2', 'Scheme 2 Additional Mass Loss']
    lcms_annotations_required_columns = ['Parent ID', 'Reject', 'Annotation', 'Class Key', 'Tail Key', 'm/z', 'RT', 'z']
    input_done = False
    while not input_done:
        param_files_pass = True
        print('')
        project_directory = input('Enter project folder directory (copy from top of file explorer): ')
        if not exists(project_directory):
            param_files_pass = False
            print('Error: could not find folder defined by user input.')
        else:
            print('')
        if param_files_pass:
            os.chdir(project_directory)
            project_directory_files = get_current_directory_files()
            has_class_definitions = False
            has_lcms_annotations = False
            for file_name in project_directory_files:
                file_name = str(file_name)
                csv_index = file_name.find('.csv')
                if csv_index > 0:
                    file_name_upper = file_name.upper()
                    class_definitions_index = file_name_upper.find('OZNOX CLASS DEFINITIONS')
                    lcms_annotations_index = file_name_upper.find('OZNOX LC-MS ANNOTATIONS')
                    if class_definitions_index >= 0:
                        has_class_definitions = True
                    elif lcms_annotations_index >= 0:
                        has_lcms_annotations = True
            if (not has_class_definitions) or (not has_lcms_annotations):
                param_files_pass = False
                print('Error: Essential .csv definitions file not found.')
                print('Project folder must contain both of the following .csv files:')
                print('OzNOx class definitions        Example: oznox class definitions_study2_04192024.csv')
                print('OzNOx LC-MS annotations        Example: Tuesday OzNOx lc-ms annotations.csv')
                print('These names are not caps-sensitive, but the spelling and spaces between the keywords matters.')
        if param_files_pass:
            print('Selecting OzNOx class definitions .csv file ...')
            for file_name in project_directory_files:
                file_name = str(file_name)
                csv_index = file_name.find('.csv')
                if csv_index > 0:
                    file_name_upper = file_name.upper()
                    class_definitions_index = file_name_upper.find('OZNOX CLASS DEFINITIONS')
                    if class_definitions_index >= 0:
                        print('OzNOx class definitions found: ' + file_name)
                        class_definitions_file = file_name
            print('Proceeding with newest OzNOx class definitions: ' + class_definitions_file)
            print('Checking OzNOx class definitions: ' + class_definitions_file)
            try:
                class_definitions_frame = pd.read_csv(class_definitions_file)
            except Exception as error_message:
                param_files_pass = False
                print(error_message)
                print('Error: formatting issue of OzNOx class definitions file.')
        if param_files_pass:
            class_definitions_frame_columns = class_definitions_frame.columns
            for required_column in class_definitions_required_columns:
                has_required_column = False
                for frame_column in class_definitions_frame_columns:
                    if required_column == frame_column:
                        has_required_column = True
                if not has_required_column:
                    param_files_pass = False
                    print('Error: ' + class_definitions_file + ' is missing column ' + required_column)
        if param_files_pass:
            class_definitions_frame_index = class_definitions_frame.index
            class_definitions_frame_size = class_definitions_frame_index.size
            class_definitions_classes = []
            if class_definitions_frame_size == 0:
                param_files_pass = False
                print('Error: OzNOx class definitions file has no entries.')
        if param_files_pass:
            for i in range(class_definitions_frame_size):
                if param_files_pass:
                    try:
                        frame_row = class_definitions_frame.loc[i]
                        row_class_key = str(frame_row['Class Key'])
                        row_num_tail_chains = int(frame_row['Num Tail Chains'])
                        row_num_fatty_acyls = int(frame_row['Num Fatty Acyls'])
                        row_num_sphingoid_bases = int(frame_row['Num Sphingoid Bases'])
                        row_oznox_scheme_1_bool = str(frame_row['OzNOx MS2 Product Ion Scheme 1'])
                        row_oznox_scheme_1_bool = row_oznox_scheme_1_bool.upper()
                        if row_oznox_scheme_1_bool == 'TRUE':
                            row_oznox_scheme_1_bool = True
                        else:
                            row_oznox_scheme_1_bool = False
                        row_oznox_scheme_2_bool = str(frame_row['OzNOx MS2 Product Ion Scheme 2'])
                        row_oznox_scheme_2_bool = row_oznox_scheme_2_bool.upper()
                        if row_oznox_scheme_2_bool == 'True':
                            row_oznox_scheme_2_bool = True
                        else:
                            row_oznox_scheme_2_bool = False
                        row_oznox_scheme_1_mass = float(frame_row['Scheme 1 Additional Mass Loss'])
                        row_oznox_scheme_2_mass = float(frame_row['Scheme 2 Additional Mass Loss'])
                    except Exception as error_message:
                        param_files_pass = False
                        print(error_message)
                        print('Error: data formatting issue in OzNOx class definitions file.')
                if param_files_pass:
                    new_class = True
                    for old_class in class_definitions_classes:
                        if row_class_key == old_class:
                            new_class = False
                    if new_class:
                        class_definitions_classes += [row_class_key]
                    else:
                        param_files_pass = False
                        print('Error: There can be only 1 entry per Class Key in OzNOx class definitions.')
                if param_files_pass:
                    if row_num_tail_chains <= 0:
                        param_files_pass = False
                        print('Error: Lipids must have at least 1 tail chain.')
                if param_files_pass:
                    if row_num_fatty_acyls < 0:
                        param_files_pass = False
                        print('Error: Lipids must have >= 0 fatty acyls.')
                if param_files_pass:
                    if row_num_sphingoid_bases < 0:
                        param_files_pass = False
                        print('Error: Lipids must have >= 0 sphingoid bases.')
                if param_files_pass:
                    if row_oznox_scheme_1_mass < 0 or row_oznox_scheme_2_mass < 0:
                        print('Warning: OzNOx MS2 Schema Additional Mass Losses will usually be >= 0.')
                if param_files_pass:
                    if not row_oznox_scheme_1_bool and not row_oznox_scheme_2_bool:
                        print('Warning: ' + str(row_class_key) + ' has no OzNOx MS2 Product Ion Schema.')
        if param_files_pass:
            print('OzNOx Class Definitions file looks good.')
            print('')
        if param_files_pass:
            print('Selecting OzNOx LC-MS annotations .csv file ...')
            for file_name in project_directory_files:
                file_name = str(file_name)
                csv_index = file_name.find('.csv')
                if csv_index > 0:
                    file_name_upper = file_name.upper()
                    lcms_annotations_index = file_name_upper.find('OZNOX LC-MS ANNOTATIONS')
                    if lcms_annotations_index >= 0:
                        print('OzNOx LC-MS annotations found: ' + file_name)
                        lcms_annotations_file = file_name
            print('Proceeding with newest OzNOx LC-MS annotations: ' + lcms_annotations_file)
            print('Checking OzNOx class definitions: ' + lcms_annotations_file)
            try:
                lcms_annotations_frame = pd.read_csv(lcms_annotations_file)
            except Exception as error_message:
                param_files_pass = False
                print(error_message)
                print('Error: formatting issue of OzNOx LC-MS annotations file.')
        if param_files_pass:
            lcms_annotations_frame_columns = lcms_annotations_frame.columns
            for required_column in lcms_annotations_required_columns:
                has_required_column = False
                for frame_column in lcms_annotations_frame_columns:
                    if required_column == frame_column:
                        has_required_column = True
                if not has_required_column:
                    param_files_pass = False
                    print('Error: ' + lcms_annotations_file + ' is missing column ' + required_column)
        if param_files_pass:
            lcms_annotations_frame_index = lcms_annotations_frame.index
            lcms_annotations_frame_size = lcms_annotations_frame_index.size
            if lcms_annotations_frame_size == 0:
                param_files_pass = False
                print('Error: OzNOx LC-MS annotations file has no entries.')
        if param_files_pass:
            unique_parent_ids = []
            unique_annotation_names = []
            for i in range(lcms_annotations_frame_size):
                if param_files_pass:
                    try:
                        frame_row = lcms_annotations_frame.loc[i]
                        row_annotation_name = str(frame_row['Annotation'])
                        row_parent_id = int(frame_row['Parent ID'])
                        row_reject = str(frame_row['Reject'])
                        row_reject = row_reject.upper()
                        if row_reject == 'FALSE':
                            row_reject = False
                        else:
                            row_reject = True
                        row_class_key = str(frame_row['Class Key'])
                        row_tail_key = str(frame_row['Tail Key'])
                        row_mz = float(frame_row['m/z'])
                        row_rt = float(frame_row['RT'])
                        row_z = float(frame_row['z'])
                    except Exception as error_message:
                        param_files_pass = False
                        print(error_message)
                        print('Error: data formatting issue in OzNOx LC-MS annotations file.')
                if param_files_pass:
                    new_parent_id = True
                    for unique_parent_id in unique_parent_ids:
                        if unique_parent_id == row_parent_id:
                            new_parent_id = False
                    if new_parent_id:
                        unique_parent_ids += [row_parent_id]
                    else:
                        param_files_pass = False
                        print('Error: All LC-MS annotations Parent IDs must be unique.')
                if param_files_pass:
                    new_annotation_name = True
                    for unique_annotation_name in unique_annotation_names:
                        if unique_annotation_name == row_annotation_name:
                            new_annotation_name = False
                    if new_annotation_name:
                        unique_annotation_names += [row_annotation_name]
                    else:
                        param_files_pass = False
                        print('Error: All LC-MS Annotation names must be unique.')
                if param_files_pass and not row_reject:
                    class_found = False
                    for class_key in class_definitions_classes:
                        if row_class_key == class_key:
                            class_found = True
                    if not class_found:
                        param_files_pass = False
                        print('Error: Non-rejected Annotations must have a Class Key in Class Definitions.')
                        print('Missing Class Definitions entry for: ' + row_class_key)
                if param_files_pass and not row_reject:
                    try:
                        row_tail_key_object = TailKey()
                        row_tail_key_object.create(row_tail_key, 100)
                    except Exception as error_message:
                        param_files_pass = False
                        print(error_message)
                        print('Error: Formatting issue of OzNOx LC-MS Annotations Tail Key(s).')
                    if not row_tail_key_object.is_valid:
                        param_files_pass = False
                        print('Error: Formatting issue of OzNOx LC-MS Annotations Tail Key(s).')
                if param_files_pass and not row_reject:
                    if row_mz <= 0:
                        param_files_pass = False
                        print('Error: OzNOx LC-MS Annotations must have m/z >= 0.')
                if param_files_pass and not row_reject:
                    if row_rt <= 0:
                        param_files_pass = False
                        print('Error: OzNOx LC-MS Annotations must have RT >= 0.')
                if param_files_pass and not row_reject:
                    if row_z <= 0 or (row_z % 1.0) > 0:
                        param_files_pass = False
                        print('Error: charge state (z) must be a positive integer.')
        if param_files_pass:
            print('OzNOx LC-MS annotations file looks good.')
            print('')
        if param_files_pass:
            input_done = True
    print('Script 1 finished.  LC-MS annotations are ready for processing')
    return project_directory, class_definitions_frame, lcms_annotations_frame


# OzNOx Script 2 inputs LC-MS annotations in .csv format and provides values for expedited manual verification/rejection
def oznox_script_2(project_directory, lcms_annotations_frame):

    os.chdir(project_directory)
    print('')
    print('Beginning Script 2 ...')
    print('')

    carbons_list = []
    doublebonds_list = []
    hydroxyls_list = []
    valid_classes = []
    reject_statuses = []
    lcms_annotations_frame_index = lcms_annotations_frame.index
    lcms_annotations_frame_size = lcms_annotations_frame_index.size

    print('Beginning Tail Key component analyses...')
    for j in range(lcms_annotations_frame_size):
        annotation_row = lcms_annotations_frame.loc[j]
        annotation_reject = str(annotation_row['Reject'])
        annotation_reject = annotation_reject.upper()
        if annotation_reject == 'FALSE':
            annotation_reject = False
        else:
            annotation_reject = True
        annotation_name = str(annotation_row['Annotation'])
        if not annotation_reject:
            annotation_class = str(annotation_row['Class Key'])
            row_tail_key = str(annotation_row['Tail Key'])
            tail_key_object = TailKey()
            tail_key_object.create(row_tail_key, 100)
            if tail_key_object.is_valid:
                doublebonds_list += [tail_key_object.num_doublebonds]
                carbons_list += [tail_key_object.num_carbons]
                hydroxyls_list += [tail_key_object.num_hydroxyls]
                new_valid_class = True
                for valid_class in valid_classes:
                    if annotation_class == valid_class:
                        new_valid_class = False
                if new_valid_class:
                    valid_classes += [annotation_class]
                reject_statuses += [False]
            else:
                print('Warning: Annotation ' + annotation_name + ' has an invalid Tail Key and has been Rejected.')
                doublebonds_list += ['']
                carbons_list += ['']
                hydroxyls_list += ['']
                reject_statuses += [True]
        else:
            doublebonds_list += ['']
            carbons_list += ['']
            hydroxyls_list += ['']
            reject_statuses += [True]
    lcms_annotations_frame['Tail Carbons'] = pd.array(carbons_list)
    temp_frame = lcms_annotations_frame.copy()
    lcms_annotations_frame = temp_frame.copy()
    lcms_annotations_frame['Tail Double-Bonds'] = pd.array(doublebonds_list)
    temp_frame = lcms_annotations_frame.copy()
    lcms_annotations_frame = temp_frame.copy()
    lcms_annotations_frame['Tail Hydroxyls'] = pd.array(hydroxyls_list)
    temp_frame = lcms_annotations_frame.copy()
    lcms_annotations_frame = temp_frame.copy()
    print('Tail Key component analyses complete.')
    print('')
    print('Beginning RT curve modeling ...')
    for valid_class in valid_classes:
        print('')
        print('On Class ' + valid_class + ' ...')
        class_indexes = []
        unique_db_counts = []
        row_carbons = []
        row_doublebonds = []
        measured_retention_times = []
        model_retention_times = []
        delta_retention_times = []
        for j in range(lcms_annotations_frame_size):
            annotation_row = lcms_annotations_frame.loc[j]
            annotation_reject = str(annotation_row['Reject'])
            annotation_reject = annotation_reject.upper()
            if annotation_reject == 'FALSE':
                annotation_reject = False
            else:
                annotation_reject = True
            annotation_class = str(annotation_row['Class Key'])
            if annotation_class == valid_class and not annotation_reject:
                row_db_count = int(annotation_row['Tail Double-Bonds'])
                new_db_count = True
                for unique_db_count in unique_db_counts:
                    if unique_db_count == row_db_count:
                        new_db_count = False
                if new_db_count:
                    unique_db_counts += [row_db_count]
                row_doublebonds += [row_db_count]
                row_carbons += [int(annotation_row['Tail Carbons'])]
                measured_retention_times += [float(annotation_row['RT'])]
                class_indexes += [j]
        if len(class_indexes) < 3:
            print('Warning: Class ' + valid_class + ' has too few entries for RT curve modeling.')
        else:
            linear_coefficients = create_2_var_linear_equation(row_carbons, row_doublebonds, measured_retention_times)
            if len(linear_coefficients) > 0:
                x0_coefficient = linear_coefficients[0]
                x1_coefficient = linear_coefficients[1]
                x2_coefficient = linear_coefficients[2]
                print('RT = ' + str(x1_coefficient) + ' * TailCarbons + ' + str(x2_coefficient) + ' * TailDBs + '
                      + str(x0_coefficient))
                for i in range(len(class_indexes)):
                    row_carbons_i = row_carbons[i]
                    row_doublebonds_i = row_doublebonds[i]
                    measured_retention_times_i = measured_retention_times[i]
                    model_retention_times_i = (x0_coefficient + x1_coefficient * row_carbons_i
                                               + x2_coefficient * row_doublebonds_i)
                    delta_retention_times_i = abs(measured_retention_times_i - model_retention_times_i)
                    model_retention_times += [model_retention_times_i]
                    delta_retention_times += [delta_retention_times_i]
            else:
                for i in range(len(class_indexes)):
                    model_retention_times += ['']
                    delta_retention_times += ['']
            for unique_db_count in unique_db_counts:
                array_indexes = []
                unique_c_counts = []
                for i in range(len(row_doublebonds)):
                    row_doublebonds_i = row_doublebonds[i]
                    if row_doublebonds_i == unique_db_count:
                        array_indexes += [i]
                        row_carbons_i = row_carbons[i]
                        new_c_count = True
                        for unique_c_count in unique_c_counts:
                            if unique_c_count == row_carbons_i:
                                new_c_count = False
                        if new_c_count:
                            unique_c_counts += [row_carbons_i]
                if len(unique_c_counts) >= 3:
                    x = []
                    y = []
                    for i in array_indexes:
                        x += [row_carbons[i]]
                        y += [measured_retention_times[i]]
                    quad_coefficients = create_1_var_quad_equation(x, y)
                    if len(quad_coefficients) > 0:
                        x2_coefficient = quad_coefficients[0]
                        x1_coefficient = quad_coefficients[1]
                        x0_coefficient = quad_coefficients[2]
                        print('For ' + str(unique_db_count) + ' DB, RT = ' + str(x2_coefficient) + ' * TailCarbons^2 + '
                              + str(x1_coefficient) + ' * TailCarbons + ' + str(x0_coefficient))
                        for i in array_indexes:
                            row_carbons_i = row_carbons[i]
                            measured_retention_times_i = measured_retention_times[i]
                            model_retention_times_i = (x2_coefficient * row_carbons_i ** 2
                                                       + x1_coefficient * row_carbons_i + x0_coefficient)
                            delta_retention_times_i = abs(measured_retention_times_i - model_retention_times_i)
                            model_retention_times[i] = model_retention_times_i
                            delta_retention_times[i] = delta_retention_times_i
            for i in range(len(class_indexes)):
                frame_index = class_indexes[i]
                model_rt = model_retention_times[i]
                delta_rt = delta_retention_times[i]
                if model_rt != '':
                    lcms_annotations_frame.loc[frame_index, 'Model RT'] = model_rt
                    lcms_annotations_frame.loc[frame_index, 'Abs (Measured RT - Modeled RT)'] = delta_rt
                    temp_frame = lcms_annotations_frame.copy()
                    lcms_annotations_frame = temp_frame.copy()

    # Reorganizing and outputting DataFrame
    frame_columns = lcms_annotations_frame.columns
    output_columns = []
    for frame_column in frame_columns:
        if frame_column != 'RT' and frame_column != 'Model RT' and frame_column != 'Abs (Measured RT - Modeled RT)':
            output_columns += [frame_column]
    output_columns += ['RT']
    output_columns += ['Model RT']
    output_columns += ['Abs (Measured RT - Modeled RT)']
    lcms_annotations_frame = lcms_annotations_frame[output_columns]
    output_name = 'OzNOx LC-MS Annotations ' + time_stamp() + '.csv'
    lcms_annotations_frame.to_csv(output_name, index=False)
    print('')
    print('Script 2 output available as ' + output_name)
    print('')
    print('Script 2 finished.  Perform manual validation/rejection and rerun Script 1 to update the program dataframe.')
    return None


# OzNOx Script 3 inputs LC-MS annotations following manual verification/rejection and creates OzNOx PRM targets
def oznox_script_3(project_directory, class_definitions_frame, lcms_annotations_frame):
    os.chdir(project_directory)
    class_definitions_frame_index = class_definitions_frame.index
    class_definitions_frame_size = class_definitions_frame_index.size
    lcms_annotations_frame_index = lcms_annotations_frame.index
    lcms_annotations_frame_size = lcms_annotations_frame_index.size

    print('')
    print('Beginning Script 3 ...')
    prm_rt_width = None
    input_done = False
    while not input_done:
        print('')
        prm_rt_width = input('Enter PRM RT window width: ')
        try:
            prm_rt_width = float(prm_rt_width)
            input_done = True
        except Exception as e:
            print(e)
            input_done = False
            print('Error: PRM RT window width must be a positive number.')
        if input_done:
            if prm_rt_width <= 0:
                input_done = False
                print('Error: PRM RT window width must be a positive number.')

    prm_mzs = []
    prm_rt_starts = []
    prm_rt_ends = []
    prm_energies = []
    for i in range(lcms_annotations_frame_size):
        annotation_row = lcms_annotations_frame.loc[i]
        annotation_reject = str(annotation_row['Reject'])
        annotation_reject = annotation_reject.upper()
        if annotation_reject == 'FALSE':
            annotation_reject = False
        else:
            annotation_reject = True
        if not annotation_reject:
            annotation_tail_key = str(annotation_row['Tail Key'])
            tail_key_object = TailKey()
            tail_key_object.create(annotation_tail_key, 100)
            if tail_key_object.is_valid and tail_key_object.num_doublebonds >= 1:
                annotation_mz = float(annotation_row['m/z'])
                annotation_class = str(annotation_row['Class Key'])
                annotation_rt = float(annotation_row['RT'])
                annotation_z = int(annotation_row['z'])
                annotation_m = annotation_mz * annotation_z
                class_energy = 15
                class_ms1_shift = 0.0
                for j in range(class_definitions_frame_size):
                    definition_row = class_definitions_frame.loc[j]
                    definition_class = str(definition_row['Class Key'])
                    if definition_class == annotation_class:
                        class_energy = float(definition_row['OzNOx MS2 Collision Energy'])
                        class_ms1_shift = float(definition_row['OzNOx MS1 m/z Shift'])
                prm_mz = (annotation_m + class_ms1_shift) / annotation_z
                prm_mzs += [prm_mz]
                prm_energies += [class_energy]
                prm_rt_starts += [(annotation_rt - (prm_rt_width / 2))]
                prm_rt_ends += [(annotation_rt + (prm_rt_width / 2))]

    prm_columns = ['Mass [m/z]', 'Formula [M]', 'Formula type', 'Species', 'CS [z]', 'Polarity', 'Start [min]',
                   'End [min]', '(N)CE', 'MSX ID', 'Comment']
    prm_frame = pd.DataFrame(index=[0], columns=prm_columns)
    for i in range(len(prm_mzs)):
        prm_mzs_i = prm_mzs[i]
        prm_rt_starts_i = prm_rt_starts[i]
        prm_rt_ends_i = prm_rt_ends[i]
        prm_energies_i = prm_energies[i]
        if i == 0:
            prm_frame.loc[i, 'Mass [m/z]'] = prm_mzs_i
            prm_frame.loc[i, 'Formula [M]'] = ''
            prm_frame.loc[i, 'Formula type'] = ''
            prm_frame.loc[i, 'Species'] = ''
            prm_frame.loc[i, 'CS [z]'] = 1
            prm_frame.loc[i, 'Polarity'] = 'Positive'
            prm_frame.loc[i, 'Start [min]'] = prm_rt_starts_i
            prm_frame.loc[i, 'End [min]'] = prm_rt_ends_i
            prm_frame.loc[i, '(N)CE'] = prm_energies_i
            prm_frame.loc[i, 'MSX ID'] = ''
            prm_frame.loc[i, 'Comment'] = ''
        else:
            data = {'Mass [m/z]': prm_mzs_i, 'Formula [M]': '', 'Formula type': '', 'Species': '', 'CS [z]': 1,
                    'Polarity': 'Positive', 'Start [min]': prm_rt_starts_i, 'End [min]': prm_rt_ends_i,
                    '(N)CE': prm_energies_i, 'MSX ID': '', 'Comment': ''}
            data_frame = pd.DataFrame(data, index=[0])
            prm_frame = pd.concat([prm_frame, data_frame], ignore_index=True)
            temp_frame = prm_frame.copy()
            prm_frame = temp_frame.copy()
    output_name = 'OzNOx PRM list ' + time_stamp() + '.csv'
    prm_frame.to_csv(output_name, index=False)
    print('')
    print('Script 3 output available as ' + output_name)
    print('')
    print('Script 3 finished.  Use the PRM list generated to conduct LC-MS2 analyses.')
    return None


# OzNOx Script 4 converts .txt LC-MS and PRM LC-MS2 data to .csv
def oznox_script_4():
    original_directory = os.getcwd()

    print('')
    print('Beginning Script 4 ...')

    txt_data_files = None
    txt_data_file_paths = None
    input_done = False
    while not input_done:
        print('')
        txt_data_directory = input('Enter LC-MS(/MS) .txt data folder directory (copy from top of file explorer): ')
        if not exists(txt_data_directory):
            print('Error: could not find directory defined by user input.')
        else:
            print('')
            os.chdir(txt_data_directory)
            txt_data_directory_files = get_current_directory_files()
            txt_data_files = []
            txt_data_file_paths = []
            for file_name in txt_data_directory_files:
                txt_index = file_name.find('.txt')
                if txt_index > 0:
                    with open(file_name, 'r') as f:
                        first_line = str(f.readline())
                        f.close()
                        msdata_index: int = first_line.find('msdata:')
                        if msdata_index >= 0:
                            print('Data found: ' + file_name)
                            txt_data_files += [file_name]
                            file_name = os.path.join(txt_data_directory, file_name)
                            txt_data_file_paths += [file_name]
            if len(txt_data_files) < 1:
                print('Error: No appropriately formatted LC-MS .txt data found.')
            else:
                input_done = True

    input_done = False
    csv_directory_files = []
    while not input_done:
        print('')
        csv_directory = input('Enter LC-MS(/MS) .csv data folder directory (copy from top of file explorer): ')
        if not exists(csv_directory):
            print('Error: could not find folder defined by user input.')
        else:
            input_done = True
            for txt_data_file in txt_data_files:
                txt_data_file = txt_data_file[:((len(txt_data_file)) - 4)]
                txt_data_file = txt_data_file + '.csv'
                file_name = os.path.join(csv_directory, txt_data_file)
                csv_directory_files += [file_name]

    for i in range(len(txt_data_files)):
        txt_file_name_i = txt_data_files[i]
        print('')
        print('Converting ' + txt_file_name_i + ' ...')
        txt_file_i = txt_data_file_paths[i]
        csv_file_i = csv_directory_files[i]
        convert_raw_txt_files_to_csv(txt_file_i, [1, 2], 0, csv_file_i)

    os.chdir(original_directory)
    print('')
    print('Script 4 finished.  The LC-MS and LC-MS2 .csv files created can be used by Script 5.')
    return None


# OzNOx Script 5 inputs LC-MS annotations and PRM OzNOx data and outputs double-bond annotations per species
def oznox_script_5(project_directory, class_definitions_frame, lcms_annotations_frame):

    mass_o16 = 15.994915
    mass_c12 = 12.0
    mass_h1 = 1.00782503226
    mass_e = 0.00054858

    print('')
    print('Beginning Script 5 ...')

    input_done = False
    data_frame = None
    csv_data_files = None
    csv_data_file_names = None
    check_ozid_aldehydes = False
    check_oznox_products = False
    csv_data_required_columns = ['Scan RT', 'MS Level', 'MSn Precursor', 'Scan Spectrum']
    while not input_done:
        print('')
        csv_directory = input('Enter LC-MS(/MS) .csv data folder directory (copy from top of file explorer): ')
        if not exists(csv_directory):
            print('Error: could not find folder defined by user input.')
        else:
            csv_data_files = []
            csv_data_file_names = []
            data_found = False
            os.chdir(csv_directory)
            directory_files = get_current_directory_files()
            print('')
            for directory_file in directory_files:
                csv_index = directory_file.find('.csv')
                if csv_index > 0:
                    try:
                        data_frame = pd.read_csv(directory_file)
                        good_frame = True
                    except Exception as error_message:
                        print(error_message)
                        print('Warning: incompatible .csv file present in the folder.')
                        good_frame = False
                    if good_frame:
                        data_frame_columns = data_frame.columns
                        has_required_columns = True
                        for required_column in csv_data_required_columns:
                            has_this_column = False
                            for frame_column in data_frame_columns:
                                if frame_column == required_column:
                                    has_this_column = True
                            if not has_this_column:
                                has_required_columns = False
                        if has_required_columns:
                            data_found = True
                            print('Found compatible .csv data: ' + directory_file)
                            csv_data_file = os.path.join(csv_directory, directory_file)
                            csv_data_files += [csv_data_file]
                            csv_data_file_names += [directory_file]
            if not data_found:
                print('Error: no compatible .csv LC-MS or LC-MS2 data found.')
            else:
                bad_length = False
                for csv_data_file_name in csv_data_file_names:
                    output_name = csv_data_file_name[:(len(csv_data_file_name) - 4)] + ' ' + time_stamp() + '.csv'
                    output_name = os.path.join(project_directory, output_name)
                    if len(output_name) > 260:
                        bad_length = True
                if bad_length:
                    print('Error: Output file path length will be too long. Try shorter folder and/or sample names.')
                else:
                    input_2_done = False
                    print('')
                    print('Enter 0 to select a different folder.')
                    print('Enter 1 to proceed with MS1 OzID aldehyde m/z detection.')
                    print('Enter 2 to proceed with MS2 OzNOx product ion detection.')
                    print('Enter 3 to proceed with both MS1 OzID aldehyde and OzNOx product ion detection.')
                    while not input_2_done:
                        print('')
                        user_choice = input('Enter 0, 1, 2, or 3: ')
                        if user_choice == '0':
                            input_2_done = True
                            check_ozid_aldehydes = False
                            check_oznox_products = False
                        elif user_choice == '1':
                            input_2_done = True
                            input_done = True
                            check_ozid_aldehydes = True
                            check_oznox_products = False
                        elif user_choice == '2':
                            input_2_done = True
                            input_done = True
                            check_ozid_aldehydes = False
                            check_oznox_products = True
                        elif user_choice == '3':
                            input_2_done = True
                            input_done = True
                            check_ozid_aldehydes = True
                            check_oznox_products = True
                        else:
                            print('')
                            print('Error: invalid user entry.')

    user_max_ozid_level = None
    ms1_mz_tolerance = None
    ms1_rt_tolerance = None
    ms2_mz_tolerance = None
    ms2_rt_tolerance = None

    if check_ozid_aldehydes:

        input_done = False
        while not input_done:

            print('')
            user_max_ozid_level = input('Enter maximum OzID reactions per molecule: ')
            try:
                user_max_ozid_level = float(user_max_ozid_level)
                good_value = True
            except Exception as error_message:
                print(error_message)
                print('Error: Maximum OzID reactions per molecule must be a positive integer.')
                good_value = False
            if good_value:
                if (user_max_ozid_level % 1.0) > 0:
                    print('Error: Maximum OzID reactions per molecule must be a positive integer.')
                    good_value = False
            if good_value:
                input_done = True
                user_max_ozid_level = int(user_max_ozid_level)

    if check_ozid_aldehydes or check_oznox_products:

        input_done = False
        while not input_done:

            print('')
            ms1_mz_tolerance = input('Enter MS1 m/z tolerance (such as 0.003): ')
            try:
                ms1_mz_tolerance = float(ms1_mz_tolerance)
                good_value = True
            except Exception as error_message:
                print(error_message)
                print('Error: MS1 m/z tolerance must be a positive number.')
                good_value = False
            if good_value:
                if ms1_mz_tolerance <= 0:
                    print('Error: MS1 m/z tolerance must be a positive number.')
                    good_value = False
            if good_value:
                input_done = True

    input_done = False
    while not input_done:
        print('')
        ms1_rt_tolerance = input('Enter MS1 RT tolerance (such as 0.10): ')
        try:
            ms1_rt_tolerance = float(ms1_rt_tolerance)
            good_value = True
        except Exception as error_message:
            print(error_message)
            print('Error: MS1 RT tolerance must be a positive number.')
            good_value = False
        if good_value:
            if ms1_rt_tolerance <= 0:
                print('Error: MS1 RT tolerance must be a positive number.')
                good_value = False
        if good_value:
            input_done = True

    if check_oznox_products:

        input_done = False
        while not input_done:
            print('')
            ms2_mz_tolerance = input('Enter MS2 m/z tolerance (such as 0.005): ')
            try:
                ms2_mz_tolerance = float(ms2_mz_tolerance)
                good_value = True
            except Exception as error_message:
                print(error_message)
                print('Error: MS2 m/z tolerance must be a positive number.')
                good_value = False
            if good_value:
                if ms2_mz_tolerance <= 0:
                    print('Error: MS2 m/z tolerance must be a positive number.')
                    good_value = False
            if good_value:
                input_done = True

        input_done = False
        while not input_done:
            print('')
            ms2_rt_tolerance = input('Enter MS2 RT tolerance (such as 0.10): ')
            try:
                ms2_rt_tolerance = float(ms2_rt_tolerance)
                good_value = True
            except Exception as error_message:
                print(error_message)
                print('Error: MS2 RT tolerance must be a positive number.')
                good_value = False
            if good_value:
                if ms2_rt_tolerance <= 0:
                    print('Error: MS2 RT tolerance must be a positive number.')
                    good_value = False
            if good_value:
                input_done = True

    print('')
    time_stamp_string = time_stamp()
    print('Beginning processing at time stamp ' + time_stamp_string)
    print('Seeking OzID aldehyde signals: ' + str(check_ozid_aldehydes))
    print('Seeking OzNOx product ion signals: ' + str(check_oznox_products))
    if check_ozid_aldehydes:
        print('Maximum OzID reactions per molecule: ' + str(user_max_ozid_level))
        print('OzID aldehyde m/z tolerance: ' + str(ms1_mz_tolerance))
        print('OzID aldehyde RT tolerance: ' + str(ms1_rt_tolerance))
    if check_oznox_products:
        print('OzNOx precursor m/z tolerance: ' + str(ms1_mz_tolerance))
        print('OzNOx precursor RT tolerance: ' + str(ms2_rt_tolerance))
        print('OzNOx product ion m/z tolerance: ' + str(ms2_mz_tolerance))

    os.chdir(project_directory)

    for i in range(len(csv_data_file_names)):

        csv_data_file = csv_data_files[i]
        csv_data_file_name = csv_data_file_names[i]
        print('')
        print('On data file ' + str(i + 1) + ' of ' + str(len(csv_data_file_names)) + ': ' + csv_data_file_name)
        print('')

        output_name = os.path.join(project_directory, csv_data_file_name)

        lcms_annotations_frame_index = lcms_annotations_frame.index
        lcms_annotations_frame_size = lcms_annotations_frame_index.size

        class_definitions_frame_index = class_definitions_frame.index
        class_definitions_frame_size = class_definitions_frame_index.size

        output_columns = ['Parent ID', 'Is Parent', 'Annotation', 'Class Key', 'Tail Key', 'Oz(NOx) Key',
                          'Scheme 2 FA Loss', 'MS Level', 'Precursor m/z', 'm/z', 'User RT', 'Max Counts RT', 'Counts',
                          'RT:Counts Spectrum']
        output_frame = pd.DataFrame(index=[0], columns=output_columns)
        first_row = True

        for j in range(lcms_annotations_frame_size):

            annotation_row = lcms_annotations_frame.loc[j]
            annotation_parent_id = annotation_row['Parent ID']
            annotation_name = str(annotation_row['Annotation'])
            print('On LC-MS annotation ' + str(j + 1) + ' of ' + str(lcms_annotations_frame_size) + ': '
                  + annotation_name)
            annotation_class_key = str(annotation_row['Class Key'])
            annotation_tail_key = str(annotation_row['Tail Key'])
            annotation_mz = float(annotation_row['m/z'])
            annotation_rt = float(annotation_row['RT'])
            annotation_z = int(annotation_row['z'])

            ms1_target_names = [annotation_name]
            ms1_target_mzs = [annotation_mz]

            class_match = False
            definition_num_chains = None
            oznox_mz_shift = None
            oznox_scheme_1 = None
            oznox_scheme_1_shift = None
            oznox_scheme_2 = None
            oznox_scheme_2_shift = None
            for k in range(class_definitions_frame_size):
                definition_row = class_definitions_frame.loc[k]
                definition_class_key = str(definition_row['Class Key'])
                if annotation_class_key == definition_class_key:
                    class_match = True
                    definition_num_chains = int(definition_row['Num Tail Chains'])
                    oznox_mz_shift = float(definition_row['OzNOx MS1 m/z Shift'])
                    oznox_scheme_1 = str(definition_row['OzNOx MS2 Product Ion Scheme 1'])
                    oznox_scheme_1 = oznox_scheme_1.upper()
                    if oznox_scheme_1 == 'TRUE':
                        oznox_scheme_1 = True
                    else:
                        oznox_scheme_1 = False
                    oznox_scheme_1_shift = float(definition_row['Scheme 1 Additional Mass Loss'])
                    oznox_scheme_2 = str(definition_row['OzNOx MS2 Product Ion Scheme 2'])
                    oznox_scheme_2 = oznox_scheme_2.upper()
                    if oznox_scheme_2 == 'TRUE':
                        oznox_scheme_2 = True
                    else:
                        oznox_scheme_2 = False
                    oznox_scheme_2_shift = float(definition_row['Scheme 2 Additional Mass Loss'])

            if class_match:

                annotation_tail_object = TailKey()
                annotation_tail_object.create(annotation_tail_key, definition_num_chains)
                is_molecular = annotation_tail_object.is_molecular_species
                num_dbs = annotation_tail_object.num_doublebonds

                if num_dbs > 0:

                    if check_ozid_aldehydes:

                        tails_with_dbs = 0
                        if is_molecular:
                            for single_chain in annotation_tail_object.cleaned_chains:
                                single_tail_object = TailKey()
                                single_tail_object.create(single_chain, 1)
                                if single_tail_object.num_doublebonds > 0:
                                    tails_with_dbs += 1
                        else:
                            tails_with_dbs = definition_num_chains

                        if tails_with_dbs > user_max_ozid_level:
                            tails_with_dbs = 1

                        if definition_num_chains > 4:
                            tails_with_dbs = 1

                        annotation_tail_object.generate_multi_ozid_aldehydes(annotation_name, annotation_mz,
                                                                             annotation_z, tails_with_dbs)
                        aldehydes = annotation_tail_object.ozid_aldehydes

                        for aldehyde in aldehydes:
                            ms1_target_names += [aldehyde[0]]
                            ms1_target_mzs += [aldehyde[1]]

                    ms1_counts_and_rts = search_csv_data(csv_data_file, 1, annotation_rt, ms1_rt_tolerance,
                                                         ms1_target_mzs, ms1_mz_tolerance, 0,
                                                         0)
                    m = 0
                    annotation_counts_and_rt = ms1_counts_and_rts[m]
                    annotation_counts = annotation_counts_and_rt[0]
                    if annotation_counts > 0:
                        annotation_rt_max = annotation_counts_and_rt[1]
                        rt_counts_spectrum = annotation_counts_and_rt[2]
                    else:
                        annotation_counts = ''
                        annotation_rt_max = ''
                        rt_counts_spectrum = ''

                    if first_row:
                        output_frame.loc[0, 'Parent ID'] = annotation_parent_id
                        output_frame.loc[0, 'Is Parent'] = 'TRUE'
                        output_frame.loc[0, 'Annotation'] = annotation_name
                        output_frame.loc[0, 'Class Key'] = annotation_class_key
                        output_frame.loc[0, 'Tail Key'] = annotation_tail_key
                        output_frame.loc[0, 'Oz(NOx) Key'] = ''
                        output_frame.loc[0, 'Scheme 2 FA Loss'] = ''
                        output_frame.loc[0, 'MS Level'] = 1
                        output_frame.loc[0, 'Precursor m/z'] = ''
                        output_frame.loc[0, 'm/z'] = annotation_mz
                        output_frame.loc[0, 'User RT'] = annotation_rt
                        output_frame.loc[0, 'Max Counts RT'] = annotation_rt_max
                        output_frame.loc[0, 'Counts'] = annotation_counts
                        output_frame.loc[0, 'RT:Counts Spectrum'] = rt_counts_spectrum
                        first_row = False
                    else:
                        data = {'Parent ID': annotation_parent_id, 'Is Parent': 'TRUE', 'Annotation': annotation_name,
                                'Class Key': annotation_class_key, 'Tail Key': annotation_tail_key, 'Oz(NOx) Key': '',
                                'Scheme 2 FA Loss': '', 'MS Level': 1, 'Precursor m/z': '', 'm/z': annotation_mz,
                                'User RT': annotation_rt, 'Max Counts RT': annotation_rt_max,
                                'Counts': annotation_counts, 'RT:Counts Spectrum': rt_counts_spectrum}
                        data_frame = pd.DataFrame(data, index=[0])
                        output_frame = pd.concat([output_frame, data_frame], ignore_index=True)
                        temp_frame = output_frame.copy()
                        output_frame = temp_frame.copy()

                    ms1_names_counts_rts = []
                    while m < (len(ms1_counts_and_rts) - 1):
                        m += 1
                        ms1_counts_and_rts_m = ms1_counts_and_rts[m]
                        row_name = ms1_target_names[m]
                        row_counts = ms1_counts_and_rts_m[0]
                        row_rt = ms1_counts_and_rts_m[1]
                        row_spectrum = ms1_counts_and_rts_m[2]
                        row_mz = ms1_target_mzs[m]
                        if row_counts > 0:
                            ms1_names_counts_rts += [[row_name, row_counts, row_rt, row_mz, row_spectrum]]
                    ms1_names_counts_rts.sort(key=lambda x: int(x[1]), reverse=True)

                    for ms1_name_counts_rt in ms1_names_counts_rts:
                        row_name = ms1_name_counts_rt[0]
                        row_counts = ms1_name_counts_rt[1]
                        row_rt = ms1_name_counts_rt[2]
                        row_mz = ms1_name_counts_rt[3]
                        row_spectrum = ms1_name_counts_rt[4]
                        row_ozid_key = create_ozid_key(row_name, annotation_name)
                        data = {'Parent ID': annotation_parent_id, 'Is Parent': 'FALSE', 'Annotation': row_name,
                                'Class Key': '', 'Tail Key': annotation_tail_key, 'Oz(NOx) Key': row_ozid_key,
                                'Scheme 2 FA Loss': '', 'MS Level': 1, 'Precursor m/z': '', 'm/z': row_mz,
                                'User RT': '', 'Max Counts RT': row_rt, 'Counts': row_counts,
                                'RT:Counts Spectrum': row_spectrum}
                        data_frame = pd.DataFrame(data, index=[0])
                        output_frame = pd.concat([output_frame, data_frame], ignore_index=True)
                        temp_frame = output_frame.copy()
                        output_frame = temp_frame.copy()

                    if check_oznox_products:

                        annotation_tail_object.generate_multi_ozid_aldehydes(annotation_name, annotation_mz,
                                                                             annotation_z, 1)
                        aldehydes = annotation_tail_object.ozid_aldehydes

                        ms2_target_names = []
                        ms2_target_mzs = []

                        if oznox_scheme_1:
                            for aldehyde in aldehydes:
                                aldehyde_name = aldehyde[0]
                                aldehyde_mz = aldehyde[1]
                                scheme_1_target_mz = aldehyde_mz - (oznox_scheme_1_shift / annotation_z)
                                ms2_target_names += [aldehyde_name]
                                ms2_target_mzs += [scheme_1_target_mz]

                        if oznox_scheme_2:
                            for aldehyde in aldehydes:
                                aldehyde_name = aldehyde[0]
                                aldehyde_mz = aldehyde[1]
                                scheme_2_target_mz_pre_loss = aldehyde_mz - (oznox_scheme_2_shift / annotation_z)
                                all_chains = annotation_tail_object.cleaned_chains
                                unique_chains = []
                                for one_chain in all_chains:
                                    new_chain = True
                                    for unique_chain in unique_chains:
                                        if unique_chain == one_chain:
                                            new_chain = False
                                    if new_chain:
                                        unique_chains += [one_chain]
                                losable_chains = []
                                for unique_chain_1 in unique_chains:
                                    single_tail_object = TailKey()
                                    single_tail_object.create(unique_chain_1, 1)
                                    is_q_species = single_tail_object.is_q_species
                                    is_p_species = single_tail_object.is_p_species
                                    is_o_species = single_tail_object.is_o_species
                                    is_d_species = single_tail_object.is_d_species
                                    is_t_species = single_tail_object.is_t_species
                                    if is_q_species or is_p_species or is_o_species or is_d_species or is_t_species:
                                        losable = False
                                    else:
                                        losable = False
                                        duplicate_found = False
                                        for one_chain in all_chains:
                                            if one_chain != unique_chain_1 and duplicate_found:
                                                single_tail_object = TailKey()
                                                single_tail_object.create(one_chain, 1)
                                                num_dbs = single_tail_object.num_doublebonds
                                                if num_dbs > 0:
                                                    losable = True
                                            elif one_chain == unique_chain_1:
                                                duplicate_found = True
                                    if losable:
                                        losable_chains += [unique_chain_1]
                                for losable_chain in losable_chains:
                                    single_tail_object = TailKey()
                                    single_tail_object.create(losable_chain, 1)
                                    scheme_2_name = aldehyde_name + ' - FA ' + str(losable_chain)
                                    ms2_target_names += [scheme_2_name]
                                    num_c = single_tail_object.num_carbons
                                    num_db = single_tail_object.num_doublebonds
                                    num_o = single_tail_object.num_hydroxyls + 2
                                    num_h = 3 + 2 * (num_c - 2) - 2 * num_db
                                    fa_mass_loss = (num_c * mass_c12 + num_o * mass_o16 + num_h * mass_h1
                                                    + 1 * mass_e)
                                    scheme_2_target_mz = scheme_2_target_mz_pre_loss - (fa_mass_loss / annotation_z)
                                    ms2_target_mzs += [scheme_2_target_mz]

                        oznox_precursor_mz = annotation_mz + (oznox_mz_shift / annotation_z)
                        ms2_counts_and_rts = search_csv_data(csv_data_file, 2, annotation_rt, ms2_rt_tolerance,
                                                             ms2_target_mzs, ms2_mz_tolerance, oznox_precursor_mz,
                                                             ms1_mz_tolerance)

                        ms2_names_counts_rts = []
                        m = -1
                        while m < (len(ms2_counts_and_rts) - 1):
                            m += 1
                            ms2_counts_and_rts_m = ms2_counts_and_rts[m]
                            row_name = ms2_target_names[m]
                            row_counts = ms2_counts_and_rts_m[0]
                            row_rt = ms2_counts_and_rts_m[1]
                            row_spectrum = ms2_counts_and_rts_m[2]
                            row_mz = ms2_target_mzs[m]
                            if row_counts > 0:
                                ms2_names_counts_rts += [[row_name, row_counts, row_rt, row_mz, row_spectrum]]
                        ms2_names_counts_rts.sort(key=lambda x: int(x[1]), reverse=True)

                        for ms2_name_counts_rt in ms2_names_counts_rts:
                            row_name = ms2_name_counts_rt[0]
                            row_counts = ms2_name_counts_rt[1]
                            row_rt = ms2_name_counts_rt[2]
                            row_mz = ms2_name_counts_rt[3]
                            row_spectrum = ms2_name_counts_rt[4]
                            row_ozid_key = create_ozid_key(row_name, annotation_name)
                            fa_loss_index = row_name.find(' - FA ')
                            if fa_loss_index > 0:
                                fa_loss = "(" + row_name[(fa_loss_index + 6):] + ")"
                            else:
                                fa_loss = ''
                            data = {'Parent ID': annotation_parent_id, 'Is Parent': 'FALSE', 'Annotation': row_name,
                                    'Class Key': '', 'Tail Key': annotation_tail_key, 'Oz(NOx) Key': row_ozid_key,
                                    'Scheme 2 FA Loss': fa_loss, 'MS Level': 2, 'Precursor m/z': oznox_precursor_mz,
                                    'm/z': row_mz, 'User RT': '', 'Max Counts RT': row_rt, 'Counts': row_counts,
                                    'RT:Counts Spectrum': row_spectrum}
                            data_frame = pd.DataFrame(data, index=[0])
                            output_frame = pd.concat([output_frame, data_frame], ignore_index=True)
                            temp_frame = output_frame.copy()
                            output_frame = temp_frame.copy()

        output_frame.to_csv(output_name, index=False)
        print('')
        print("Script 5 output available as: " + str(output_name))

    print('')
    print('Script 5 finished.  The newly created files in the project folder can be used for double-bond annotation.')
    return None


# OzNOx Script 6 combines the Script 5 .csv outputs into a single .csv file
def oznox_script_6(lcms_annotations_frame):

    print('')
    print('Beginning Script 6 ...')

    lcms_annotations_frame_index = lcms_annotations_frame.index
    lcms_annotations_frame_size = lcms_annotations_frame_index.size
    lcms_annotation_names_ids = []
    for i in range(lcms_annotations_frame_size):
        frame_row = lcms_annotations_frame.loc[i]
        frame_row_annotation = frame_row['Annotation']
        frame_row_parent_id = frame_row['Parent ID']
        lcms_annotation_names_ids += [[frame_row_annotation, frame_row_parent_id]]

    input_done = False
    data_frame = None
    csv_data_files = None
    csv_data_file_names = None
    row_is_parent = None
    row_parent_id = None
    row_annotation = None
    csv_data_required_columns = ['Parent ID', 'Is Parent', 'Annotation', 'Class Key', 'Tail Key', 'Oz(NOx) Key',
                                 'Scheme 2 FA Loss', 'MS Level', 'Precursor m/z', 'm/z', 'User RT', 'Max Counts RT',
                                 'Counts', 'RT:Counts Spectrum']
    while not input_done:
        print('')
        csv_directory = input('Enter folder with Script 5 .csv outputs (copy from top of file explorer): ')
        if not exists(csv_directory):
            print('Error: could not find folder defined by user input.')
        else:
            csv_data_files = []
            csv_data_file_names = []
            data_found = False
            os.chdir(csv_directory)
            directory_files = get_current_directory_files()
            print('')
            for directory_file in directory_files:
                csv_index = directory_file.find('.csv')
                if csv_index > 0:
                    try:
                        data_frame = pd.read_csv(directory_file)
                        good_frame = True
                    except Exception as error_message:
                        print(error_message)
                        print('Warning: incompatible .csv file present in the folder.')
                        good_frame = False
                    if good_frame:
                        data_frame_columns = data_frame.columns
                        has_required_columns = True
                        for required_column in csv_data_required_columns:
                            has_this_column = False
                            for frame_column in data_frame_columns:
                                if frame_column == required_column:
                                    has_this_column = True
                            if not has_this_column:
                                has_required_columns = False
                        if has_required_columns:
                            data_frame_index = data_frame.index
                            data_frame_size = data_frame_index.size
                            parent_ids = []
                            child_ids = []
                            unique_parent_annotations = []
                            for i in range(data_frame_size):
                                if good_frame:
                                    try:
                                        frame_row = data_frame.loc[i]
                                        row_parent_id = int(frame_row['Parent ID'])
                                        row_is_parent = str(frame_row['Is Parent'])
                                        row_annotation = str(frame_row['Annotation'])
                                        row_is_parent_upper = row_is_parent.upper()
                                        if row_is_parent_upper == 'TRUE':
                                            row_is_parent = True
                                        elif row_is_parent_upper == 'FALSE':
                                            row_is_parent = False
                                        else:
                                            good_frame = False
                                            print('Error: Is Parent must either be TRUE or FALSE.')
                                    except Exception as error_message:
                                        good_frame = False
                                        print(error_message)
                                        print('Error: Formatting issue of script 5 output: ' + directory_file)
                                if good_frame and row_is_parent:
                                    match_found = False
                                    search_finished = False
                                    i = -1
                                    while not search_finished:
                                        i += 1
                                        if i == len(lcms_annotation_names_ids):
                                            search_finished = True
                                        else:
                                            lcms_annotation_name = lcms_annotation_names_ids[i][0]
                                            if row_annotation == lcms_annotation_name:
                                                match_found = True
                                                search_finished = True
                                    if not match_found:
                                        good_frame = False
                                        print('Error: Missing LC-MS annotation for Parent: ' + str(row_annotation))
                                if good_frame:
                                    if row_is_parent:
                                        new_parent = True
                                        for parent_id in parent_ids:
                                            if parent_id == row_parent_id:
                                                new_parent = False
                                        if not new_parent:
                                            good_frame = False
                                            print('Error: There can be only one Is Parent per Parent ID per .csv')
                                        else:
                                            parent_ids += [row_parent_id]
                                    else:
                                        new_child = True
                                        for child_id in child_ids:
                                            if child_id == row_parent_id:
                                                new_child = False
                                        if new_child:
                                            child_ids += [row_parent_id]
                                if good_frame and row_is_parent:
                                    new_parent_annotation = True
                                    for unique_parent_annotation in unique_parent_annotations:
                                        if unique_parent_annotation == row_annotation:
                                            new_parent_annotation = False
                                    if new_parent_annotation:
                                        unique_parent_annotations += [row_annotation]
                                    else:
                                        good_frame = False
                                        print('Error: duplicate Parent Annotations Names found: ' + row_annotation)
                            if good_frame:
                                for child_id in child_ids:
                                    if good_frame:
                                        match_found = False
                                        search_finished = False
                                        i = -1
                                        while not search_finished:
                                            i += 1
                                            if i == len(parent_ids):
                                                search_finished = True
                                            else:
                                                parent_id = parent_ids[i]
                                                if child_id == parent_id:
                                                    match_found = True
                                                    search_finished = True
                                        if not match_found:
                                            good_frame = False
                                            print('Error: missing Is Parent Annotation for Parent ID: ' + str(child_id))
                            if good_frame:
                                data_found = True
                                print('Found compatible .csv data: ' + directory_file)
                                csv_data_files += [directory_file]
                                csv_data_file_name = directory_file[:(len(directory_file) - 4)]
                                csv_data_file_names += [csv_data_file_name]
            if data_found:
                input_done_2 = False
                while not input_done_2:
                    print('')
                    user_input = input('Proceed? Enter Y for Yes or N for No: ')
                    if user_input == 'Y' or user_input == 'y':
                        input_done_2 = True
                        input_done = True
                    elif user_input == 'N' or user_input == 'n':
                        input_done_2 = True
                    else:
                        print('Error: Invalid user input.  Enter Y or N.')
            else:
                print('Error: no compatible .csv LC-MS or LC-MS2 data found.')

    csv_data_files.sort()
    csv_data_file_names.sort()

    print('')
    print('Importing data ...')
    print('')

    output_columns = ['Parent ID', 'Is Parent', 'Annotation', 'Class Key', 'Tail Key', 'Oz(NOx) Key',
                      'Scheme 2 FA Loss', 'MS Level', 'Precursor m/z', 'm/z', 'User RT', 'All Samples Max Counts']
    for csv_data_file_name in csv_data_file_names:
        rt_column_name = 'Max Counts RT (' + csv_data_file_name + ')'
        counts_column_name = 'Counts (' + csv_data_file_name + ')'
        output_columns += [rt_column_name]
        output_columns += [counts_column_name]
    for csv_data_file_name in csv_data_file_names:
        spectrum_column_name = 'RT:Counts Spectrum (' + csv_data_file_name + ')'
        output_columns += [spectrum_column_name]
    output_frame = pd.DataFrame(index=[0], columns=output_columns)

    data_rows = []
    unique_rows = []
    for i in range(len(csv_data_files)):
        print('Importing data from file ' + str(i + 1) + ' of ' + str(len(csv_data_files)) + ' ...')
        csv_data_file = csv_data_files[i]
        csv_data_file_name = csv_data_file_names[i]
        data_frame = pd.read_csv(csv_data_file)
        temp_frame = data_frame.sort_values(by=['Parent ID', 'Is Parent', 'Counts'], ascending=[True, False, False])
        data_frame = temp_frame.copy()
        data_frame_index = data_frame.index
        data_frame_size = data_frame_index.size
        parent_id_swaps = []
        for j in range(data_frame_size):
            data_frame_row = data_frame.loc[j]
            data_frame_row_is_parent = str(data_frame_row['Is Parent'])
            data_frame_row_is_parent = data_frame_row_is_parent.upper()
            if data_frame_row_is_parent == 'TRUE':
                data_frame_row_is_parent = True
            else:
                data_frame_row_is_parent = False
            if data_frame_row_is_parent:
                data_frame_row_annotation = str(data_frame_row['Annotation'])
                data_frame_row_parent_id = int(data_frame_row['Parent ID'])
                search_finished = False
                ref_parent_id = None
                k = -1
                while not search_finished:
                    k += 1
                    ref_parent_annotation = lcms_annotation_names_ids[k][0]
                    if ref_parent_annotation == data_frame_row_annotation:
                        search_finished = True
                        ref_parent_id = lcms_annotation_names_ids[k][1]
                if data_frame_row_parent_id != ref_parent_id:
                    parent_id_swaps += [[data_frame_row_parent_id, ref_parent_id]]
        if len(parent_id_swaps) > 0:
            sorted_swaps = sorted(parent_id_swaps, key=lambda x: x[1], reverse=True)
            for parent_id_swap in sorted_swaps:
                old_id = parent_id_swap[0]
                final_id = parent_id_swap[1]

                below_target = False
                j = len(data_frame_size) - 1
                while not below_target:
                    if j == 0 or j == -1:
                        below_target = True
                    else:
                        data_frame_row = data_frame.loc[j]
                        data_frame_row_parent_id = int(data_frame_row['Parent ID'])
                        if data_frame_row_parent_id < old_id:
                            below_target = True
                        else:
                            j = float(j)
                            j = int(0.80 * j)
                if j < 0:
                    j = 0
                j -= 1
                above_target = False
                found_target = False
                while not above_target:
                    j += 1
                    if j >= (data_frame_size - 1):
                        above_target = True
                    else:
                        data_frame_row = data_frame.loc[j]
                        data_frame_row_parent_id = int(data_frame_row['Parent ID'])
                        if data_frame_row_parent_id == old_id:
                            data_frame.loc[j, 'Parent ID'] = final_id
                            found_target = True
                        elif found_target:
                            above_target = True

        for j in range(data_frame_size):
            data_frame_row = data_frame.loc[j]
            data_frame_row_parent_id = int(data_frame_row['Parent ID'])
            data_frame_row_is_parent = str(data_frame_row['Is Parent'])
            data_frame_row_annotation = str(data_frame_row['Annotation'])
            data_frame_row_class_key = str(data_frame_row['Class Key'])
            if data_frame_row_class_key.upper() == 'NAN':
                data_frame_row_class_key = ''
            data_frame_row_tail_key = str(data_frame_row['Tail Key'])
            if data_frame_row_tail_key.upper() == 'NAN':
                data_frame_row_tail_key = ''
            data_frame_row_oznox_key = str(data_frame_row['Oz(NOx) Key'])
            if data_frame_row_oznox_key.upper() == 'NAN':
                data_frame_row_oznox_key = ''
            data_frame_row_scheme_2_loss = str(data_frame_row['Scheme 2 FA Loss'])
            if data_frame_row_scheme_2_loss.upper() == 'NAN':
                data_frame_row_scheme_2_loss = ''
            data_frame_row_ms_level = int(data_frame_row['MS Level'])
            data_frame_row_precursor_mz = str(data_frame_row['Precursor m/z'])
            if data_frame_row_precursor_mz.upper() == 'NAN':
                data_frame_row_precursor_mz = ''
            else:
                data_frame_row_precursor_mz = float(data_frame_row_precursor_mz)
            data_frame_row_mz = str(data_frame_row['m/z'])
            if data_frame_row_mz.upper() == 'NAN':
                data_frame_row_mz = ''
            else:
                data_frame_row_mz = float(data_frame_row_mz)
            data_frame_row_user_rt = str(data_frame_row['User RT'])
            if data_frame_row_user_rt.upper() == 'NAN':
                data_frame_row_user_rt = ''
            else:
                data_frame_row_user_rt = float(data_frame_row_user_rt)
            data_frame_row_max_counts_rt = str(data_frame_row['Max Counts RT'])
            if data_frame_row_max_counts_rt.upper() == 'NAN':
                data_frame_row_max_counts_rt = ''
            else:
                data_frame_row_max_counts_rt = float(data_frame_row_max_counts_rt)
            data_frame_row_counts = str(data_frame_row['Counts'])
            if data_frame_row_counts.upper() == 'NAN':
                data_frame_row_counts = 0.0
            else:
                data_frame_row_counts = float(data_frame_row_counts)
            data_frame_row_spectrum = str(data_frame_row['RT:Counts Spectrum'])
            data_row = (csv_data_file_name, data_frame_row_annotation, data_frame_row_class_key,
                        data_frame_row_tail_key, data_frame_row_oznox_key, data_frame_row_scheme_2_loss,
                        data_frame_row_ms_level, data_frame_row_precursor_mz, data_frame_row_mz, data_frame_row_user_rt,
                        data_frame_row_max_counts_rt, data_frame_row_counts, data_frame_row_spectrum,
                        data_frame_row_parent_id, data_frame_row_is_parent)
            data_rows += [data_row]

            new_annotation = True
            for unique_row in unique_rows:
                unique_row_name = unique_row[1]
                unique_row_ms_level = unique_row[6]
                if unique_row_name == data_frame_row_annotation and unique_row_ms_level == data_frame_row_ms_level:
                    new_annotation = False
            if new_annotation:
                unique_rows += [data_row]

    sorted_data_rows = sorted(data_rows, key=lambda x: (x[1], x[6], x[0]), reverse=False)
    sorted_unique_rows = sorted(unique_rows, key=lambda x: (x[1], x[6]), reverse=False)

    print('')
    print('Merging data ...')
    print('')

    output_index = -1
    last_percent = -1
    for i in range(len(sorted_unique_rows)):

        output_index += 1
        unique_row = sorted_unique_rows[i]

        current_percent = int(100 * ((i + 1)/len(sorted_unique_rows)))
        if current_percent != last_percent:
            last_percent = current_percent
            print('Merging is ' + str(current_percent) + '% finished')

        output_annotation = unique_row[1]
        output_class_key = unique_row[2]
        output_tail_key = unique_row[3]
        output_oznox_key = unique_row[4]
        output_scheme_2_loss = unique_row[5]
        output_ms_level = unique_row[6]
        output_precursor_mz = unique_row[7]
        output_mz = unique_row[8]
        output_user_rt = unique_row[9]
        output_parent_id = unique_row[13]
        output_is_parent = unique_row[14]

        if output_index == 0:

            output_frame.loc[output_index, 'Parent ID'] = output_parent_id
            output_frame.loc[output_index, 'Is Parent'] = output_is_parent
            output_frame.loc[output_index, 'Annotation'] = output_annotation
            output_frame.loc[output_index, 'Class Key'] = output_class_key
            output_frame.loc[output_index, 'Tail Key'] = output_tail_key
            output_frame.loc[output_index, 'Oz(NOx) Key'] = output_oznox_key
            output_frame.loc[output_index, 'Scheme 2 FA Loss'] = output_scheme_2_loss
            output_frame.loc[output_index, 'MS Level'] = output_ms_level
            output_frame.loc[output_index, 'Precursor m/z'] = output_precursor_mz
            output_frame.loc[output_index, 'm/z'] = output_mz
            output_frame.loc[output_index, 'User RT'] = output_user_rt
            output_frame.loc[output_index, 'All Samples Max Counts'] = 0.0
            for csv_data_file_name in csv_data_file_names:
                rt_column_name = 'Max Counts RT (' + csv_data_file_name + ')'
                counts_column_name = 'Counts (' + csv_data_file_name + ')'
                spectrum_column_name = 'RT:Counts Spectrum (' + csv_data_file_name + ')'
                output_frame.loc[output_index, rt_column_name] = ''
                output_frame.loc[output_index, counts_column_name] = ''
                output_frame.loc[output_index, spectrum_column_name] = ''

        else:

            data = {'Parent ID': output_parent_id, 'Is Parent': output_is_parent, 'Annotation': output_annotation,
                    'Class Key': output_class_key, 'Tail Key': output_tail_key, 'Oz(NOx) Key': output_oznox_key,
                    'Scheme 2 FA Loss': output_scheme_2_loss, 'MS Level': output_ms_level,
                    'Precursor m/z': output_precursor_mz, 'm/z': output_mz, 'User RT': output_user_rt,
                    'All Samples Max Counts': 0.0}
            for csv_data_file_name in csv_data_file_names:
                rt_column_name = 'Max Counts RT (' + csv_data_file_name + ')'
                counts_column_name = 'Counts (' + csv_data_file_name + ')'
                spectrum_column_name = 'RT:Counts Spectrum (' + csv_data_file_name + ')'
                data[rt_column_name] = ''
                data[counts_column_name] = ''
                data[spectrum_column_name] = ''
            new_frame = pd.DataFrame(data=data, index=[0])
            temp_frame = pd.concat([output_frame, new_frame], ignore_index=True)
            output_frame = temp_frame.copy()

        below_target = False
        j = len(sorted_data_rows) - 1
        while not below_target:
            if j == 0 or j == -1:
                below_target = True
            else:
                sorted_data_row = sorted_data_rows[j]
                sorted_data_row_annotation = str(sorted_data_row[1])
                if sorted_data_row_annotation != output_annotation:
                    two_annotations = [output_annotation, sorted_data_row_annotation]
                    two_annotations.sort()
                    first_in_alphabet = two_annotations[0]
                    if first_in_alphabet == sorted_data_row_annotation:
                        below_target = True
                    else:
                        j = float(j)
                        j = int(0.80 * j)
                else:
                    j = float(j)
                    j = int(0.80 * j)

        if j < 0:
            j = 0
        j -= 1

        above_target = False
        found_target = False
        target_indexes = []
        while not above_target:
            j += 1
            if j >= (len(sorted_data_rows) - 1):
                above_target = True
            else:
                sorted_data_row = sorted_data_rows[j]
                sorted_data_row_annotation = sorted_data_row[1]
                sorted_data_row_ms_level = sorted_data_row[6]
                if sorted_data_row_annotation == output_annotation and sorted_data_row_ms_level == output_ms_level:
                    target_indexes += [j]
                    found_target = True
                elif found_target:
                    above_target = True

        for j in target_indexes:

            old_max_counts = output_frame.loc[output_index, 'All Samples Max Counts']
            sorted_data_row = sorted_data_rows[j]
            csv_data_file_name = sorted_data_row[0]
            sorted_data_row_max_counts_rt = sorted_data_row[10]
            sorted_data_row_counts = str(sorted_data_row[11])
            if len(sorted_data_row_counts) == 0 or sorted_data_row_counts.upper() == 'NAN':
                sorted_data_row_counts = ''
            else:
                sorted_data_row_counts = float(sorted_data_row_counts)
                if sorted_data_row_counts > old_max_counts:
                    output_frame.loc[output_index, 'All Samples Max Counts'] = sorted_data_row_counts
                elif sorted_data_row_counts == 0:
                    sorted_data_row_counts = ''
            sorted_data_row_spectrum = str(sorted_data_row[12])
            if len(sorted_data_row_spectrum) == 0 or sorted_data_row_spectrum.upper() == 'NAN':
                sorted_data_row_spectrum = ''
            rt_column_name = 'Max Counts RT (' + csv_data_file_name + ')'
            counts_column_name = 'Counts (' + csv_data_file_name + ')'
            spectrum_column_name = 'RT:Counts Spectrum (' + csv_data_file_name + ')'
            output_frame.loc[output_index, rt_column_name] = sorted_data_row_max_counts_rt
            output_frame.loc[output_index, counts_column_name] = sorted_data_row_counts
            output_frame.loc[output_index, spectrum_column_name] = sorted_data_row_spectrum

    print('')
    print('Organizing merged data ...')
    print('')

    temp_frame = output_frame.sort_values(by=['Parent ID', 'Is Parent', 'MS Level', 'All Samples Max Counts'],
                                          ascending=[True, False, True, False])
    output_frame = temp_frame.copy()

    output_name = 'OzNOx Script 6 output ' + time_stamp() + '.csv'
    output_frame.to_csv(output_name, index=False)
    print('Output available as: ' + output_name)
    print('')
    print('Script 6 finished.  See the newly created .csv file combining your prior Script 5 outputs.')
    return None


# check_db_annotation() checks if a set of OzID keys satisfies the requirements of its tail double-bond counts
# returns True or False
def check_db_annotation(tail_db_counts, codex_num_chains, accepted_keys):

    tail_db_counts.sort(key=None, reverse=True)
    is_molecular = True
    highest_db_loss = 0
    if len(tail_db_counts) < codex_num_chains:
        is_molecular = False
    n_numbers_by_loss_dbs = []
    for ozid_key in accepted_keys:
        ozid_key_values = read_ozid_key(ozid_key)
        n_number = ozid_key_values[0]
        loss_dbs = ozid_key_values[1]
        if loss_dbs > highest_db_loss:
            highest_db_loss = loss_dbs
        if len(n_numbers_by_loss_dbs) < (loss_dbs + 1):
            finished_expanding = False
            while not finished_expanding:
                if len(n_numbers_by_loss_dbs) == (loss_dbs + 1):
                    finished_expanding = True
                else:
                    n_numbers_by_loss_dbs += [[]]
        prior_set = n_numbers_by_loss_dbs[loss_dbs]
        prior_set += [n_number]
        n_numbers_by_loss_dbs[loss_dbs] = prior_set
    previous_length = -1
    for n_numbers_by_loss_db in n_numbers_by_loss_dbs:
        current_length = len(n_numbers_by_loss_db)
        if previous_length != -1:
            if current_length > previous_length:
                return False
        previous_length = current_length
    db_loss_counts = []
    for n_numbers_by_loss_db in n_numbers_by_loss_dbs:
        db_loss_count = len(n_numbers_by_loss_db)
        db_loss_counts += [db_loss_count]
    if is_molecular:
        expect_neutral_loss_db_counts = []
        for i in range(highest_db_loss+1):
            expect_neutral_loss_db_counts += [0]
        for tail_db_count in tail_db_counts:
            finished_tallying = False
            while not finished_tallying:
                if tail_db_count == 0:
                    finished_tallying = True
                else:
                    tail_db_count -= 1
                    current_pos_count = expect_neutral_loss_db_counts[tail_db_count]
                    expect_neutral_loss_db_counts[tail_db_count] = (current_pos_count + 1)
        for j in range(len(expect_neutral_loss_db_counts)):
            expected_count = expect_neutral_loss_db_counts[j]
            if len(db_loss_counts) >= (j+1):
                actual_count = db_loss_counts[j]
            else:
                return False
            if actual_count != expected_count:
                return False
        theoretical_chains = []
        for j in range(codex_num_chains):
            theoretical_chain = []
            for i in range(tail_db_counts[j]):
                theoretical_chain += [-1]
            theoretical_chains += [theoretical_chain]
        n_numbers_by_loss_dbs.reverse()
        for j in range(len(n_numbers_by_loss_dbs)):
            n_number_set = n_numbers_by_loss_dbs[j]
            n_number_set.sort(key=None, reverse=True)
            if len(n_number_set) > codex_num_chains:
                return False
            for n_number in n_number_set:
                placed = False
                for i in range(len(theoretical_chains)):
                    theoretical_chain = theoretical_chains[i]
                    if len(theoretical_chain) > 0 and not placed:
                        q = (highest_db_loss + 1) - len(theoretical_chain)
                        if not placed and len(theoretical_chain) >= (j-q+1):
                            if j > 0:
                                if (j-1-q) >= 0:
                                    previous_value = theoretical_chain[j-1-q]
                                    location_value = theoretical_chain[j-q]
                                    if (((previous_value == -1) or (previous_value >= (n_number + 2)))
                                            and (location_value == -1)):
                                        theoretical_chains[i][j-q] = n_number
                                        placed = True
                                elif (j-q) >= 0:
                                    location_value = theoretical_chain[j-q]
                                    if location_value == -1:
                                        theoretical_chains[i][j-q] = n_number
                                        placed = True
                            elif j == 0 and q == 0:
                                location_value = theoretical_chain[j-q]
                                if location_value == -1:
                                    theoretical_chains[i][j-q] = n_number
                                    placed = True
                if not placed:
                    return False
        for theoretical_chain in theoretical_chains:
            previous_n = -1
            for current_n in theoretical_chain:
                if previous_n != -1:
                    if current_n > (previous_n - 2):
                        return False
                previous_n = current_n
    else:
        theoretical_chains = []
        for j in range(codex_num_chains):
            theoretical_chain = []
            for i in range(highest_db_loss+1):
                theoretical_chain += [-1]
            theoretical_chains += [theoretical_chain]
        for j in range(len(n_numbers_by_loss_dbs)):
            n_number_set = n_numbers_by_loss_dbs[j]
            n_number_set.sort(key=None, reverse=True)
            if len(n_number_set) > codex_num_chains:
                return False
            for n_number in n_number_set:
                placed = False
                for i in range(len(theoretical_chains)):
                    theoretical_chain = theoretical_chains[i]
                    if not placed and len(theoretical_chain) >= (j+1):
                        if j > 0:
                            previous_value = theoretical_chain[j-1]
                            location_value = theoretical_chain[j]
                            if (((previous_value == -1) or (previous_value >= (n_number + 2)))
                                    and (location_value == -1)):
                                theoretical_chains[i][j] = n_number
                                placed = True
                        else:
                            location_value = theoretical_chain[j]
                            if location_value == -1:
                                theoretical_chains[i][j] = n_number
                                placed = True
                if not placed:
                    return False
        for theoretical_chain in theoretical_chains:
            previous_n = -1
            for current_n in theoretical_chain:
                if previous_n != -1:
                    if current_n > (previous_n - 2):
                        return False
                    if current_n == -1:
                        return False
                previous_n = current_n
    return True


# get_db_pos_combos() returns an array of arrays containing possible double-bond n-# combinations based on OzID keys
def get_db_pos_combos(annotation_tail_key, codex_num_chains, accepted_keys):

    valid_db_positions = []
    tail_key_object = TailKey()
    tail_key_object.create(annotation_tail_key, codex_num_chains)
    tail_db_counts = []
    tail_chains = tail_key_object.cleaned_chains
    for tail_chain in tail_chains:
        tail_chain_object = TailKey()
        tail_chain_object.create(tail_chain, 1)
        tail_chain_dbs = tail_chain_object.num_doublebonds
        if tail_chain_object.is_p_species:
            tail_chain_dbs -= 1
        tail_db_counts += [tail_chain_dbs]
    is_molecular = True
    if len(tail_db_counts) < codex_num_chains:
        is_molecular = False
    tail_n_num_limits = []
    for i in range(len(tail_chains)):
        tail_chain = tail_chains[i]
        tail_chain_object = TailKey()
        tail_chain_object.create(tail_chain, 1)
        tail_length = tail_chain_object.num_carbons
        if tail_chain_object.is_t_species or tail_chain_object.is_d_species:
            tail_length -= 4
        elif tail_chain_object.is_p_species:
            tail_length -= 4
        elif tail_chain_object.is_o_species:
            tail_length -= 3
        else:
            tail_length -= 2
        if is_molecular:
            tail_n_num_limits += [tail_length]
        else:
            if len(tail_chains) == 1:
                for j in range(codex_num_chains):
                    tail_n_num_limits += [tail_length-3]
            else:
                if i == 0:
                    other_tails_count = 0
                    for j in range(len(tail_chains)):
                        if j != 0:
                            other_tails_count += 1
                    tail_length -= (3 * (codex_num_chains-1-other_tails_count))
                    for j in range(codex_num_chains-other_tails_count):
                        tail_n_num_limits += [tail_length]
    n_numbers_by_loss_dbs = []
    highest_db_loss = 0
    for ozid_key in accepted_keys:
        ozid_key_values = read_ozid_key(ozid_key)
        n_number = ozid_key_values[0]
        loss_dbs = ozid_key_values[1]
        if loss_dbs > highest_db_loss:
            highest_db_loss = loss_dbs
        if len(n_numbers_by_loss_dbs) < (loss_dbs + 1):
            finished_expanding = False
            while not finished_expanding:
                if len(n_numbers_by_loss_dbs) == (loss_dbs + 1):
                    finished_expanding = True
                else:
                    n_numbers_by_loss_dbs += [[]]
        prior_set = n_numbers_by_loss_dbs[loss_dbs]
        prior_set += [n_number]
        n_numbers_by_loss_dbs[loss_dbs] = prior_set
    theoretical_chains_template = []
    if is_molecular:
        for j in range(codex_num_chains):
            theoretical_chain = []
            for i in range(tail_db_counts[j]):
                theoretical_chain += [-1]
            theoretical_chains_template += [theoretical_chain]
    else:
        for j in range(codex_num_chains):
            theoretical_chain = []
            for i in range(highest_db_loss+1):
                theoretical_chain += [-1]
            theoretical_chains_template += [theoretical_chain]
    theoretical_chains_builds = []
    for j in range(len(n_numbers_by_loss_dbs)):
        n_number_set = n_numbers_by_loss_dbs[j]
        for n_number in n_number_set:
            if len(theoretical_chains_builds) == 0:
                theoretical_chains = copy.deepcopy(theoretical_chains_template)
                for i in range(len(theoretical_chains)):
                    theoretical_chain = theoretical_chains[i]
                    if len(theoretical_chain) > 0:
                        theoretical_chains[i][0] = n_number
                        theoretical_chains_builds += [theoretical_chains]
                        theoretical_chains = copy.deepcopy(theoretical_chains_template)
            else:
                prior_builds = copy.deepcopy(theoretical_chains_builds)
                new_builds = []
                for theoretical_chains_build in prior_builds:
                    theoretical_chains = copy.deepcopy(theoretical_chains_build)
                    for i in range(len(theoretical_chains)):
                        theoretical_chain = theoretical_chains[i]
                        if len(theoretical_chain) >= (j+1):
                            current_value = theoretical_chain[j]
                            if current_value == -1:
                                theoretical_chains[i][j] = n_number
                                new_builds += [theoretical_chains]
                                theoretical_chains = copy.deepcopy(theoretical_chains_build)
                theoretical_chains_builds = new_builds
    unique_builds = []
    for theoretical_chains_build in theoretical_chains_builds:
        novel_build = True
        for unique_build in unique_builds:
            if theoretical_chains_build == unique_build:
                novel_build = False
        if novel_build:
            unique_builds += [theoretical_chains_build]
    for unique_build in unique_builds:
        good_build = True
        for i in range(len(unique_build)):
            chain = unique_build[i]
            highest_value = 0
            previous_value = -1
            hit_end = False
            for current_value in chain:
                if current_value != -1 and hit_end:
                    good_build = False
                if current_value < (previous_value+2):
                    if current_value != -1:
                        good_build = False
                previous_value = current_value
                if current_value > highest_value:
                    highest_value = current_value
                if highest_value > tail_n_num_limits[i]:
                    good_build = False
        if good_build and not is_molecular:
            temp_arrays = []
            for chain in unique_build:
                temp_array = []
                for current_value in chain:
                    if current_value != -1:
                        temp_array += [current_value]
                temp_arrays += [temp_array]
            valid_db_positions += [temp_arrays]
        elif good_build:
            valid_db_positions += [unique_build]
    return valid_db_positions


# recursive_db_combo_builder recursively builds sets of theoretical chain-db combinations
# A non-theoretical example: 16:0/18:1/20:2 would be represented here as [0, 1, 2]
# a non-molecular species with 3 unknown chains and 1 assumed DB would return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
def recursive_db_combo_builder(prior_chains, target_length, target_sum):

    new_chains = []
    last_round = False
    for prior_chain in prior_chains:
        prior_length = len(prior_chain)
        if prior_length == (target_length - 1):
            last_round = True
        prior_sum = 0
        for prior_element in prior_chain:
            prior_sum += prior_element
        new_element = -1
        finished = False
        while not finished:
            new_element += 1
            new_sum = prior_sum + new_element
            if new_sum > target_sum:
                finished = True
            else:
                new_chain = copy.deepcopy(prior_chain)
                new_chain += [new_element]
                new_chains += [new_chain]
    if last_round:
        final_chains = []
        for new_chain in new_chains:
            final_sum = 0
            for final_element in new_chain:
                final_sum += final_element
            if final_sum == target_sum:
                final_chains += [new_chain]
        return final_chains
    else:
        return recursive_db_combo_builder(new_chains, target_length, target_sum)


# recursive_oznox_combo_builder recursively builds set of theoretical Oz(NOx) Key combinations using the reported
# Oz(NOx) Keys
def recursive_oznox_combo_builder(individual_tail_key, prior_combos, unique_keys, is_molecular, codex_num_chains):

    tail_chain_object = TailKey()
    tail_chain_object.create(individual_tail_key, 1)
    tail_length = tail_chain_object.num_carbons
    if tail_chain_object.is_t_species or tail_chain_object.is_d_species:
        tail_length -= 4
    elif tail_chain_object.is_p_species:
        tail_length -= 4
    elif tail_chain_object.is_o_species:
        tail_length -= 3
    else:
        tail_length -= 2
    if not is_molecular:
        tail_length -= (3 * (codex_num_chains - 1))

    new_combos = []
    last_round = True
    made_new_combo = False
    for prior_combo in prior_combos:
        prior_element = prior_combo[(len(prior_combo) - 1)]
        oznox_key_values = read_ozid_key(prior_element)
        n_position = oznox_key_values[0]
        neutral_loss_db = oznox_key_values[1]
        neutral_loss_oh = oznox_key_values[2]
        if neutral_loss_db > 0:
            last_round = False
            for unique_oznox_key in unique_keys:
                oznox_key_values_i = read_ozid_key(unique_oznox_key)
                n_position_i = oznox_key_values_i[0]
                neutral_loss_db_i = oznox_key_values_i[1]
                neutral_loss_oh_i = oznox_key_values_i[2]
                if (n_position_i <= (n_position - 2) and neutral_loss_db_i == (neutral_loss_db - 1)
                        and neutral_loss_oh_i <= neutral_loss_oh):
                    made_new_combo = True
                    new_combo = copy.deepcopy(prior_combo)
                    new_combo += [unique_oznox_key]
                    new_combos += [new_combo]
    if last_round:
        final_combos = []
        for new_combo in prior_combos:
            if len(final_combos) == 0:
                final_combos += [new_combo]
            else:
                unique_combo = True
                for final_combo in final_combos:
                    matches = 0
                    for i in range(len(final_combo)):
                        final_combo_element = final_combo[i]
                        new_combo_element = new_combo[i]
                        if new_combo_element == final_combo_element:
                            matches += 1
                    if matches == len(final_combo):
                        unique_combo = False
                if unique_combo:
                    final_combos += [new_combo]
        return final_combos
    elif made_new_combo:
        return recursive_oznox_combo_builder(individual_tail_key, new_combos, unique_keys, is_molecular,
                                             codex_num_chains)
    else:
        return []


# Descending sorts and returns an array of OzNOx Keys on by loss DBs, then neutral loss OH, then n-#
def sort_oznox_keys(oznox_keys):

    output_array = []
    while len(oznox_keys) > 0:
        leader_index = -1
        highest_db = -1
        highest_oh = -1
        highest_n = -1
        for i in range(len(oznox_keys)):
            oznox_key = oznox_keys[i]
            key_values = read_ozid_key(oznox_key)
            db = key_values[1]
            oh = key_values[2]
            n_pos = key_values[0]
            if leader_index == -1:
                leader_index = i
                highest_db = db
                highest_oh = oh
                highest_n = n_pos
            else:
                if db > highest_db:
                    leader_index = i
                    highest_db = db
                    highest_oh = oh
                    highest_n = n_pos
                elif db == highest_db and oh > highest_oh:
                    leader_index = i
                    highest_db = db
                    highest_oh = oh
                    highest_n = n_pos
                elif db == highest_db and oh == highest_oh and n_pos > highest_n:
                    leader_index = i
                    highest_db = db
                    highest_oh = oh
                    highest_n = n_pos
        output_array += [oznox_keys[leader_index]]
        temp_array = []
        for i in range(len(oznox_keys)):
            if i != leader_index:
                temp_array += [oznox_keys[i]]
        oznox_keys = temp_array
    return output_array


def convert_spectrum_string_to_array(spectrum_string):

    if len(spectrum_string) == 0 or spectrum_string == '0:0' or spectrum_string.upper() == 'NAN':
        return [[0.0, 0.0]]
    else:
        return_array = []
        spectrum_elements = spectrum_string.split(' ')
        for spectrum_element in spectrum_elements:
            spectrum_element_pieces = spectrum_element.split(':')
            spectrum_element_rt = float(spectrum_element_pieces[0])
            spectrum_element_value = float(spectrum_element_pieces[1])
            return_array += [[spectrum_element_rt, spectrum_element_value]]
    return return_array


def align_spectrum_to_rt_values(spectrum_array, master_rt_values):

    aligned_spectrum = []

    for c in range(len(master_rt_values)):
        if c > 0:
            previous_master_rt_value = master_rt_values[(c - 1)]
            has_previous_rt = True
        else:
            previous_master_rt_value = -1
            has_previous_rt = False
        current_master_rt_value = master_rt_values[c]
        if c <= (len(master_rt_values) - 2):
            next_master_rt_value = master_rt_values[(c + 1)]
            has_next_rt = True
        else:
            has_next_rt = False
            next_master_rt_value = -1
        if has_next_rt and has_previous_rt:
            rt_diff_cutoff_1 = abs(current_master_rt_value - previous_master_rt_value) / 2.0
            rt_diff_cutoff_2 = abs(current_master_rt_value - next_master_rt_value) / 2.0
            if rt_diff_cutoff_1 < rt_diff_cutoff_2:
                master_rt_cutoff = rt_diff_cutoff_1
            else:
                master_rt_cutoff = rt_diff_cutoff_2
        elif has_next_rt:
            master_rt_cutoff = abs(next_master_rt_value - current_master_rt_value) / 2.0
        else:
            master_rt_cutoff = abs(current_master_rt_value - previous_master_rt_value) / 2.0

        # searching through product_spectrum_i
        f = -1
        found_rt = False
        finished_scanning = False
        closest_f_index = 0
        closest_index_rt_diff = -1
        while f < (len(spectrum_array) - 1) and not finished_scanning:
            f += 1
            spectrum_element = spectrum_array[f]
            spectrum_element_rt = spectrum_element[0]
            rt_diff_i = abs(current_master_rt_value - spectrum_element_rt)
            if closest_index_rt_diff == -1 or rt_diff_i < closest_index_rt_diff:
                closest_f_index = f
                closest_index_rt_diff = rt_diff_i
            if rt_diff_i <= master_rt_cutoff:
                found_rt = True
                closest_f_index = f
            if found_rt and rt_diff_i > closest_index_rt_diff:
                finished_scanning = True
        if found_rt:
            spectrum_element = spectrum_array[closest_f_index]
            spectrum_element_counts = spectrum_element[1]
        else:
            spectrum_element_counts = 0.0
        aligned_spectrum += [[current_master_rt_value, spectrum_element_counts]]

    return aligned_spectrum


def find_unique_key_set_in_set_of_key_sets(set_of_key_sets):

    remaining_sets = copy.deepcopy(set_of_key_sets)
    unique_key_set = []

    if len(remaining_sets) == 1:
        return [remaining_sets[0], []]

    search_finished = False
    for i in range(len(remaining_sets)):
        if not search_finished:
            key_set_i = set_of_key_sets[i]
            has_unique_key = False
            for key_i in key_set_i:
                key_is_unique = True
                j = 0
                while key_is_unique and j < len(remaining_sets):
                    if j != i:
                        key_set_j = set_of_key_sets[j]
                        for key_j in key_set_j:
                            if key_i == key_j:
                                key_is_unique = False
                    j += 1
                if key_is_unique:
                    has_unique_key = True
            if has_unique_key:
                search_finished = True
                remaining_sets.pop(i)
                unique_key_set = key_set_i

    return [unique_key_set, remaining_sets]


def is_set_of_key_sets_solvable(set_of_key_sets):

    remaining_sets = copy.deepcopy(set_of_key_sets)
    while len(remaining_sets) > 1:
        found_unique_set = False
        for i in range(len(remaining_sets)):
            if not found_unique_set:
                key_set_i = remaining_sets[i]
                has_unique_key = False
                for key_i in key_set_i:
                    key_is_unique = True
                    j = 0
                    while key_is_unique and j < len(remaining_sets):
                        if j != i:
                            key_set_j = remaining_sets[j]
                            for key_j in key_set_j:
                                if key_i == key_j:
                                    key_is_unique = False
                        j += 1
                    if key_is_unique:
                        has_unique_key = True
                if has_unique_key:
                    found_unique_set = True
                    remaining_sets.pop(i)
        if not found_unique_set:
            return False
    return True


def get_top_solvable_key_sets(scores_and_key_sets):

    scores_and_key_sets.sort(key=lambda x: x[0], reverse=True)
    max_score = scores_and_key_sets[0][0]
    num_scores = len(scores_and_key_sets)
    hit_1_percent = False
    finished = False
    scores_and_key_set_index = -1
    top_solvable_key_sets = []
    while not finished:
        new_top_solvable_key_sets = copy.deepcopy(top_solvable_key_sets)
        scores_and_key_set_index += 1
        if scores_and_key_set_index == num_scores:
            finished = True
        else:
            score_and_key_set_i = scores_and_key_sets[scores_and_key_set_index]
            score_i = score_and_key_set_i[0]
            if (max_score - score_i) > 2.0:
                hit_1_percent = True
            key_set_i = score_and_key_set_i[1]
            new_top_solvable_key_sets += [key_set_i]
            if len(new_top_solvable_key_sets) == 1:
                top_solvable_key_sets = copy.deepcopy(new_top_solvable_key_sets)
            elif is_set_of_key_sets_solvable(new_top_solvable_key_sets):
                top_solvable_key_sets = copy.deepcopy(new_top_solvable_key_sets)
            else:
                if hit_1_percent:
                    return top_solvable_key_sets
                else:
                    return []
    return top_solvable_key_sets


# OzNOx Script 7 utilizes Script 6's output to assign double-bond positions
def oznox_script_7():

    print('')
    print('Beginning Script 7 ...')

    input_done = False
    sample_definitions_file = None
    sample_definitions_frame = None
    sample_definitions_frame_size = None
    row_sample_name = None
    row_ozid_bool = None
    row_oznox_bool = None
    row_sample_group = None
    sample_definitions_samples = None
    sample_definitions_groups = None
    unique_groups_and_ozid_samples = None
    unique_groups_and_only_ozid_samples = None
    unique_groups_and_oznox_samples = None
    unique_groups_and_only_oznox_samples = None
    unique_groups_and_ozid_oznox_bools = None
    script_6_output_file = None
    script_6_output_frame = None
    script_6_output_frame_size = None
    max_counts_rt_column_names = None
    counts_column_names = None
    spectrum_column_names = None
    row_annotation = None
    row_oznox_key = None
    row_ms_level = None
    row_mz = None
    row_is_parent = None
    row_parent_id = None
    parent_ids = None
    class_definitions_file = None
    class_definitions_frame = None
    class_definitions_frame_size = None
    row_class_key = None
    row_num_tail_chains = None
    row_num_fatty_acyls = None
    row_num_sphingoid_bases = None
    class_definitions_classes = None
    using_ozid = False
    sample_definitions_required_columns = ['Sample Name', 'OzID (MS1)', 'OzNOx (MS2)', 'Sample Group']
    base_script_6_output_required_columns = ['Parent ID', 'Is Parent', 'Annotation', 'Class Key', 'Tail Key',
                                             'Oz(NOx) Key', 'Scheme 2 FA Loss', 'MS Level', 'Precursor m/z', 'm/z',
                                             'User RT']
    class_definitions_required_columns = ['Class Key', 'Num Tail Chains', 'Num Fatty Acyls', 'Num Sphingoid Bases',
                                          'OzNOx MS1 m/z Shift', 'OzNOx MS2 Collision Energy',
                                          'OzNOx MS2 Product Ion Scheme 1', 'Scheme 1 Additional Mass Loss',
                                          'OzNOx MS2 Product Ion Scheme 2', 'Scheme 2 Additional Mass Loss']
    # User inputs, loading files
    while not input_done:

        print('')
        print('Script 7 requires three .csv files: Script 6 output, sample definitions, and class definitions.')
        csv_directory = input('Enter folder with the necessary .csv files (copy from top of file explorer): ')
        if not exists(csv_directory):
            print('Error: could not find folder defined by user input.')
        else:
            param_files_pass = True
            os.chdir(csv_directory)
            csv_directory_files = get_current_directory_files()
            has_class_definitions = False
            has_script_6_output = False
            has_sample_definitions = False
            for file_name in csv_directory_files:
                file_name = str(file_name)
                csv_index = file_name.find('.csv')
                if csv_index > 0:
                    file_name_upper = file_name.upper()
                    class_definitions_index = file_name_upper.find('OZNOX CLASS DEFINITIONS')
                    sample_definitions_index = file_name_upper.find('OZNOX SAMPLE DEFINITIONS')
                    script_6_output_index = file_name_upper.find('OZNOX SCRIPT 6 OUTPUT')
                    if class_definitions_index >= 0:
                        has_class_definitions = True
                    elif sample_definitions_index >= 0:
                        has_sample_definitions = True
                    elif script_6_output_index >= 0:
                        has_script_6_output = True
            if (not has_class_definitions) or (not has_sample_definitions) or (not has_script_6_output):
                param_files_pass = False
                print('Error: Essential .csv not found.')
                print('Project folder must contain all three of the following .csv files:')
                print('OzNOx class definitions        Example: oznox class definitions_study2_04192024.csv')
                print('OzNOx sample definitions       Example: Tuesday OzNOx sample definitions.csv')
                print('OzNOx Script 6 output          Example: OzNOx Script 6 output 2024_07_23_10_23_51.csv')
                print('These names are not caps-sensitive, but the spelling and spaces between the keywords matters.')

            # Selecting, checking class definitions
            if param_files_pass:
                print('')
                print('Selecting OzNOx class definitions .csv file ...')
                for file_name in csv_directory_files:
                    file_name = str(file_name)
                    csv_index = file_name.find('.csv')
                    if csv_index > 0:
                        file_name_upper = file_name.upper()
                        class_definitions_index = file_name_upper.find('OZNOX CLASS DEFINITIONS')
                        if class_definitions_index >= 0:
                            print('OzNOx class definitions found: ' + file_name)
                            class_definitions_file = file_name
                print('Proceeding with newest OzNOx class definitions: ' + class_definitions_file)
                print('Checking OzNOx class definitions: ' + class_definitions_file)
                try:
                    class_definitions_frame = pd.read_csv(class_definitions_file)
                except Exception as error_message:
                    param_files_pass = False
                    print(error_message)
                    print('Error: formatting issue of OzNOx class definitions file.')
            if param_files_pass:
                class_definitions_frame_columns = class_definitions_frame.columns
                for required_column in class_definitions_required_columns:
                    has_required_column = False
                    for frame_column in class_definitions_frame_columns:
                        if required_column == frame_column:
                            has_required_column = True
                    if not has_required_column:
                        param_files_pass = False
                        print('Error: ' + class_definitions_file + ' is missing column ' + required_column)
            if param_files_pass:
                class_definitions_frame_index = class_definitions_frame.index
                class_definitions_frame_size = class_definitions_frame_index.size
                class_definitions_classes = []
                if class_definitions_frame_size == 0:
                    param_files_pass = False
                    print('Error: OzNOx class definitions file has no entries.')
            if param_files_pass:
                for i in range(class_definitions_frame_size):
                    if param_files_pass:
                        try:
                            frame_row = class_definitions_frame.loc[i]
                            row_class_key = str(frame_row['Class Key'])
                            row_num_tail_chains = int(frame_row['Num Tail Chains'])
                            row_num_fatty_acyls = int(frame_row['Num Fatty Acyls'])
                            row_num_sphingoid_bases = int(frame_row['Num Sphingoid Bases'])
                        except Exception as error_message:
                            param_files_pass = False
                            print(error_message)
                            print('Error: data formatting issue in OzNOx class definitions file.')
                    if param_files_pass:
                        new_class = True
                        for old_class in class_definitions_classes:
                            if row_class_key == old_class:
                                new_class = False
                        if new_class:
                            class_definitions_classes += [row_class_key]
                        else:
                            param_files_pass = False
                            print('Error: There can be only 1 entry per Class Key in OzNOx class definitions.')
                    if param_files_pass:
                        if row_num_tail_chains <= 0:
                            param_files_pass = False
                            print('Error: Lipids must have >= 1 tail chain.')
                    if param_files_pass:
                        if row_num_fatty_acyls < 0:
                            param_files_pass = False
                            print('Error: Lipids must have >= 0 fatty acyls.')
                    if param_files_pass:
                        if row_num_sphingoid_bases < 0:
                            param_files_pass = False
                            print('Error: Lipids must have >= 0 sphingoid bases.')
            if param_files_pass:
                print('OzNOx Class Definitions file looks good.')
                print('')

            # Selecting, checking sample definitions
            if param_files_pass:
                print('Selecting OzNOx sample definitions .csv file ...')
                for file_name in csv_directory_files:
                    file_name = str(file_name)
                    csv_index = file_name.find('.csv')
                    if csv_index > 0:
                        file_name_upper = file_name.upper()
                        sample_definitions_index = file_name_upper.find('OZNOX SAMPLE DEFINITIONS')
                        if sample_definitions_index >= 0:
                            print('OzNOx sample definitions found: ' + file_name)
                            sample_definitions_file = file_name
                print('Proceeding with newest OzNOx sample definitions: ' + sample_definitions_file)
                print('Checking OzNOx sample definitions: ' + sample_definitions_file)
                try:
                    sample_definitions_frame = pd.read_csv(sample_definitions_file)
                except Exception as error_message:
                    param_files_pass = False
                    print(error_message)
                    print('Error: formatting issue of OzNOx sample definitions file.')
            if param_files_pass:
                sample_definitions_frame_columns = sample_definitions_frame.columns
                for required_column in sample_definitions_required_columns:
                    has_required_column = False
                    for frame_column in sample_definitions_frame_columns:
                        if required_column == frame_column:
                            has_required_column = True
                    if not has_required_column:
                        param_files_pass = False
                        print('Error: ' + sample_definitions_file + ' is missing column ' + required_column)
            if param_files_pass:
                sample_definitions_frame_index = sample_definitions_frame.index
                sample_definitions_frame_size = sample_definitions_frame_index.size
                sample_definitions_samples = []
                sample_definitions_groups = []
                if sample_definitions_frame_size == 0:
                    param_files_pass = False
                    print('Error: OzNOx sample definitions file has no entries.')
            if param_files_pass:
                for i in range(sample_definitions_frame_size):
                    if param_files_pass:
                        try:
                            frame_row = sample_definitions_frame.loc[i]
                            row_sample_name = str(frame_row['Sample Name'])
                            csv_index = row_sample_name.find('.csv')
                            if csv_index > 0:
                                row_sample_name = row_sample_name[:(len(row_sample_name) - 4)]
                            row_ozid_bool = str(frame_row['OzID (MS1)'])
                            row_oznox_bool = str(frame_row['OzNOx (MS2)'])
                            row_sample_group = str(frame_row['Sample Group'])
                        except Exception as error_message:
                            param_files_pass = False
                            print(error_message)
                            print('Error: data formatting issue in OzNOx sample definitions file.')
                    if param_files_pass:
                        row_sample_name_upper = row_sample_name.upper()
                        if row_sample_name_upper == 'NAN' or len(row_sample_name) == 0:
                            param_files_pass = False
                            print('Error: OzNOx sample definitions Sample Names cannot be blank or \'nan\'.')
                    if param_files_pass:
                        row_sample_group_upper = row_sample_group.upper()
                        if row_sample_group_upper == 'NAN' or len(row_sample_group) == 0:
                            param_files_pass = False
                            print('Error: OzNOx sample definitions Sample Group cannot be blank or \'nan\'.')
                    if param_files_pass:
                        new_sample = True
                        for old_sample in sample_definitions_samples:
                            if row_sample_name == old_sample:
                                new_sample = False
                        if new_sample:
                            sample_definitions_samples += [row_sample_name]
                        else:
                            param_files_pass = False
                            print('Error: There can be only 1 entry per Sample Name in OzNOx sample definitions.')
                    if param_files_pass:
                        row_ozid_bool = row_ozid_bool.upper()
                        if row_ozid_bool == 'TRUE':
                            row_ozid_bool = True
                        elif row_ozid_bool == 'FALSE':
                            row_ozid_bool = False
                        else:
                            param_files_pass = False
                            print('Error: OzNOx sample definitions OzID (MS1) must be TRUE or FALSE.')
                    if param_files_pass:
                        row_oznox_bool = row_oznox_bool.upper()
                        if row_oznox_bool == 'TRUE':
                            row_oznox_bool = True
                        elif row_oznox_bool == 'FALSE':
                            row_oznox_bool = False
                        else:
                            param_files_pass = False
                            print('Error: OzNOx sample definitions OzNOx (MS2) must be TRUE or FALSE.')
                    if param_files_pass:
                        if row_num_sphingoid_bases < 0:
                            param_files_pass = False
                            print('Error: Lipids must have >= 0 sphingoid bases.')
                    if param_files_pass:
                        if not (row_ozid_bool or row_oznox_bool):
                            print('Warning: Sample Name ' + row_sample_name + ' is FALSE for both OzID and OzNOx.')
                    if param_files_pass:
                        new_group = True
                        for old_group in sample_definitions_groups:
                            if row_sample_group == old_group:
                                new_group = False
                        if new_group:
                            sample_definitions_groups += [row_sample_group]
            if param_files_pass:
                groups_and_samples = []
                unique_groups_and_samples = []
                unique_groups_and_ozid_samples = []
                unique_groups_and_only_ozid_samples = []
                unique_groups_and_oznox_samples = []
                unique_groups_and_only_oznox_samples = []
                for i in range(sample_definitions_frame_size):
                    frame_row = sample_definitions_frame.loc[i]
                    row_sample_name = str(frame_row['Sample Name'])
                    row_ozid_bool = str(frame_row['OzID (MS1)'])
                    row_ozid_bool = row_ozid_bool.upper()
                    if row_ozid_bool == 'TRUE':
                        row_ozid_bool = True
                    elif row_ozid_bool == 'FALSE':
                        row_ozid_bool = False
                    row_oznox_bool = str(frame_row['OzNOx (MS2)'])
                    row_oznox_bool = row_oznox_bool.upper()
                    if row_oznox_bool == 'TRUE':
                        row_oznox_bool = True
                    elif row_oznox_bool == 'FALSE':
                        row_oznox_bool = False
                    row_sample_group = str(frame_row['Sample Group'])
                    groups_and_samples += [[row_sample_group, row_sample_name, row_ozid_bool, row_oznox_bool]]
                for sample_definitions_group in sample_definitions_groups:
                    unique_group_and_samples = [sample_definitions_group, []]
                    unique_group_and_ozid_samples = [sample_definitions_group, []]
                    unique_group_and_only_ozid_samples = [sample_definitions_group, []]
                    unique_group_and_oznox_samples = [sample_definitions_group, []]
                    unique_group_and_only_oznox_samples = [sample_definitions_group, []]
                    for group_and_sample in groups_and_samples:
                        group_i = group_and_sample[0]
                        sample_i = group_and_sample[1]
                        ozid_i = group_and_sample[2]
                        oznox_i = group_and_sample[3]
                        if group_i == sample_definitions_group:
                            samples = unique_group_and_samples[1]
                            samples += [sample_i]
                            unique_group_and_samples[1] = samples
                            if ozid_i:
                                samples = unique_group_and_ozid_samples[1]
                                samples += [sample_i]
                                unique_group_and_ozid_samples[1] = samples
                                if not oznox_i:
                                    samples = unique_group_and_only_ozid_samples[1]
                                    samples += [sample_i]
                                    unique_group_and_only_ozid_samples[1] = samples
                            if oznox_i:
                                samples = unique_group_and_oznox_samples[1]
                                samples += [sample_i]
                                unique_group_and_oznox_samples[1] = samples
                                if not ozid_i:
                                    samples = unique_group_and_only_oznox_samples[1]
                                    samples += [sample_i]
                                    unique_group_and_only_oznox_samples[1] = samples
                    unique_groups_and_samples += [unique_group_and_samples]
                    unique_groups_and_ozid_samples += [unique_group_and_ozid_samples]
                    unique_groups_and_only_ozid_samples += [unique_group_and_only_ozid_samples]
                    unique_groups_and_oznox_samples += [unique_group_and_oznox_samples]
                    unique_groups_and_only_oznox_samples += [unique_group_and_only_oznox_samples]
            if param_files_pass:
                print('OzNOx sample definitions file looks good.')
                print('')

            # Selecting, checking Script 6 output
            if param_files_pass:
                print('Selecting OzNOx Script 6 output .csv file ...')
                for file_name in csv_directory_files:
                    file_name = str(file_name)
                    csv_index = file_name.find('.csv')
                    if csv_index > 0:
                        file_name_upper = file_name.upper()
                        script_6_output_index = file_name_upper.find('OZNOX SCRIPT 6 OUTPUT')
                        if script_6_output_index >= 0:
                            print('OzNOx Script 6 output found: ' + file_name)
                            script_6_output_file = file_name
                print('Proceeding with newest OzNOx Script 6 output: ' + script_6_output_file)
                print('Checking OzNOx Script 6 output: ' + script_6_output_file)
                try:
                    script_6_output_frame = pd.read_csv(script_6_output_file)
                except Exception as error_message:
                    param_files_pass = False
                    print(error_message)
                    print('Error: formatting issue of OzNOx Script 6 output file.')
            if param_files_pass:
                max_counts_rt_column_names = []
                counts_column_names = []
                spectrum_column_names = []
                script_6_output_frame_columns = script_6_output_frame.columns
                script_6_output_required_columns = base_script_6_output_required_columns
                for sample_definitions_sample in sample_definitions_samples:
                    rt_column_name = 'Max Counts RT (' + sample_definitions_sample + ')'
                    counts_column_name = 'Counts (' + sample_definitions_sample + ')'
                    spectrum_column_name = 'RT:Counts Spectrum (' + sample_definitions_sample + ')'
                    script_6_output_required_columns += [rt_column_name]
                    script_6_output_required_columns += [counts_column_name]
                    script_6_output_required_columns += [spectrum_column_name]
                    max_counts_rt_column_names += [rt_column_name]
                    counts_column_names += [counts_column_name]
                    spectrum_column_names += [spectrum_column_name]
                for required_column in script_6_output_required_columns:
                    has_required_column = False
                    for frame_column in script_6_output_frame_columns:
                        if required_column == frame_column:
                            has_required_column = True
                    if not has_required_column:
                        param_files_pass = False
                        print('Error: ' + script_6_output_file + ' is missing column ' + required_column)
            if param_files_pass:
                script_6_output_frame_index = script_6_output_frame.index
                script_6_output_frame_size = script_6_output_frame_index.size
                if script_6_output_frame_size == 0:
                    param_files_pass = False
                    print('Error: OzNOx Script 6 output file has no entries.')
            if param_files_pass:
                script_6_annotations_ms_levels = []
                parent_ids = []
                child_ids = []
                for i in range(script_6_output_frame_size):
                    row_rts = []
                    row_counts_s = []
                    row_spectra = []
                    if param_files_pass:
                        try:
                            frame_row = script_6_output_frame.loc[i]
                            row_parent_id = int(frame_row['Parent ID'])
                            row_is_parent = str(frame_row['Is Parent'])
                            row_is_parent_upper = row_is_parent.upper()
                            if row_is_parent_upper == 'TRUE':
                                row_is_parent = True
                            elif row_is_parent_upper == 'FALSE':
                                row_is_parent = False
                            else:
                                param_files_pass = False
                                print('Error: Is Parent must either be TRUE or FALSE.')
                            row_annotation = str(frame_row['Annotation'])
                            row_class_key = str(frame_row['Class Key'])
                            if len(row_class_key) == 0 or row_class_key.upper() == 'NAN':
                                row_class_key = ''
                            row_tail_key = str(frame_row['Tail Key'])
                            row_tail_key_object = TailKey()
                            row_tail_key_object.create(row_tail_key, 100)
                            if not row_tail_key_object.is_valid:
                                param_files_pass = False
                                print('Error: Tail Key invalid: ' + row_tail_key)
                            row_oznox_key = str(frame_row['Oz(NOx) Key'])
                            if len(row_oznox_key) == 0 or row_oznox_key.upper() == 'NAN':
                                row_oznox_key = ''
                            row_scheme_2_loss = str(frame_row['Scheme 2 FA Loss'])
                            if not (len(row_scheme_2_loss) == 0 or row_scheme_2_loss.upper() == 'NAN'):
                                row_scheme_2_loss_object = TailKey()
                                row_scheme_2_loss_object.create(row_scheme_2_loss, 1)
                                if not row_scheme_2_loss_object.is_valid:
                                    param_files_pass = False
                                    print('Error: Scheme 2 FA Loss invalid: ' + row_scheme_2_loss)
                            row_ms_level = int(frame_row['MS Level'])
                            row_precursor_mz = str(frame_row['Precursor m/z'])
                            if len(row_precursor_mz) > 0 and row_precursor_mz.upper() != 'NAN':
                                row_precursor_mz = float(row_precursor_mz)
                                if row_precursor_mz <= 0:
                                    param_files_pass = False
                                    print('Error: Precursor m/z can be left blank but otherwise must be > 0.')
                            row_mz = float(frame_row['m/z'])
                            row_user_rt = str(frame_row['User RT'])
                            if len(row_user_rt) > 0 and row_user_rt.upper() != 'NAN':
                                row_user_rt = float(row_user_rt)
                                if row_user_rt <= 0:
                                    param_files_pass = False
                                    print('Error: User RT can be left blank but otherwise must be > 0.')
                            for max_counts_rt_column_name in max_counts_rt_column_names:
                                row_max_counts_rt = str(frame_row[max_counts_rt_column_name])
                                if not (len(row_max_counts_rt) == 0 or row_max_counts_rt.upper() == 'NAN'):
                                    row_max_counts_rt = float(row_max_counts_rt)
                                    row_rts += [row_max_counts_rt]
                                    if row_max_counts_rt < 0:
                                        param_files_pass = False
                                        print('Error: Max Counts RT must be blank or >= 0.')
                            for counts_column_name in counts_column_names:
                                row_counts = str(frame_row[counts_column_name])
                                if not (len(row_counts) == 0 or row_counts.upper() == 'NAN'):
                                    row_counts = float(row_counts)
                                    row_counts_s += [row_counts]
                                    if row_counts < 0:
                                        param_files_pass = False
                                        print('Error: Counts must be blank or >= 0.')
                            for spectrum_column_name in spectrum_column_names:
                                row_spectrum = str(frame_row[spectrum_column_name])
                                if not (len(row_spectrum) == 0 or row_spectrum.upper() == 'NAN'):
                                    row_spectra += [row_spectrum]
                        except Exception as error_message:
                            param_files_pass = False
                            print(error_message)
                            print('Error: data formatting issue in OzNOx Script 6 output file.')
                    if param_files_pass:
                        if row_is_parent:
                            new_parent = True
                            for parent_id in parent_ids:
                                if parent_id == row_parent_id:
                                    new_parent = False
                            if not new_parent:
                                param_files_pass = False
                                print('Error: There can be only one Parent Annotation per Parent ID.')
                            else:
                                parent_ids += [row_parent_id]
                        else:
                            new_child = True
                            for child_id in child_ids:
                                if child_id == row_parent_id:
                                    new_child = False
                            if new_child:
                                child_ids += [row_parent_id]
                    if param_files_pass:
                        for child_id in child_ids:
                            if param_files_pass:
                                search_done = False
                                match_found = False
                                j = -1
                                while not search_done:
                                    j += 1
                                    if j == len(parent_ids):
                                        search_done = True
                                    else:
                                        parent_id = parent_ids[j]
                                        if parent_id == child_id:
                                            search_done = True
                                            match_found = True
                                if not match_found:
                                    param_files_pass = False
                                    print('Error: missing Is Parent Annotation for Parent ID: ' + str(child_id))
                    if param_files_pass:
                        if row_annotation.upper() == 'NAN' or len(row_annotation) == 0:
                            param_files_pass = False
                            print('Error: OzNOx Script 6 output Annotations cannot be blank or \'nan\'.')
                        else:
                            new_annotation = True
                            for old_annotation in script_6_annotations_ms_levels:
                                old_annotation_name = old_annotation[0]
                                old_annotation_ms_level = old_annotation[1]
                                if old_annotation_name == row_annotation and old_annotation_ms_level == row_ms_level:
                                    param_files_pass = False
                                    new_annotation = False
                                    print('Error: Duplicate annotation found: ' + row_annotation + ' MS Level '
                                          + str(row_ms_level))
                            if new_annotation:
                                script_6_annotations_ms_levels += [[row_annotation, row_ms_level]]
                    if param_files_pass:
                        if len(row_class_key) > 0:
                            search_done = False
                            found_match = False
                            j = -1
                            while not search_done:
                                j += 1
                                if j == len(class_definitions_classes):
                                    search_done = True
                                else:
                                    defined_class = class_definitions_classes[j]
                                    if defined_class == row_class_key:
                                        search_done = True
                                        found_match = True
                            if not found_match:
                                param_files_pass = False
                                print('Error: Class definitions entry missing for class: ' + row_class_key)
                    if param_files_pass:
                        if len(row_oznox_key) > 0:
                            key_interpretation = read_ozid_key(row_oznox_key)
                            if len(key_interpretation) == 0:
                                param_files_pass = False
                                print('Error: Invalid Oz(NOx) key: ' + row_oznox_key)
                    if param_files_pass:
                        if row_ms_level != 1 and row_ms_level != 2:
                            param_files_pass = False
                            print('Error: Invalid MS Level: ' + str(row_ms_level))
                    if param_files_pass:
                        if (not row_is_parent) and len(row_oznox_key) == 0:
                            param_files_pass = False
                            print('Error: FALSE Is Parent Annotations must have an Oz(NOx) Key')
                    if param_files_pass:
                        if row_mz <= 0:
                            param_files_pass = False
                            print('Error: m/z must be > 0.')
                    if param_files_pass:
                        if len(row_rts) != len(row_counts_s) == 0 or len(row_spectra) != len(row_rts):
                            param_files_pass = False
                            print('Error: RT, counts data fragmented for: ' + str(row_annotation))
            if param_files_pass:
                print('OzNOx Script 6 output file looks good.')

            # Reporting which sample data will be used per group
            if param_files_pass:
                using_ozid = False
                unique_groups_and_ozid_oznox_bools = []
                for sample_definitions_group in sample_definitions_groups:
                    has_oznox_only = False
                    has_oznox = False
                    has_ozid_only = False
                    has_ozid = False
                    ref_samples_1 = None
                    ref_samples_2 = None
                    ref_samples_3 = None
                    ref_samples_4 = None
                    for unique_group_and_only_oznox_samples in unique_groups_and_only_oznox_samples:
                        ref_group = unique_group_and_only_oznox_samples[0]
                        ref_samples_1 = unique_group_and_only_oznox_samples[1]
                        if ref_group == sample_definitions_group and len(ref_samples_1) > 0:
                            has_oznox = True
                            has_oznox_only = True
                    if not has_oznox:
                        for unique_group_and_oznox_samples in unique_groups_and_oznox_samples:
                            ref_group = unique_group_and_oznox_samples[0]
                            ref_samples_2 = unique_group_and_oznox_samples[1]
                            if ref_group == sample_definitions_group and len(ref_samples_2) > 0:
                                has_oznox = True
                    for unique_group_and_only_ozid_samples in unique_groups_and_only_ozid_samples:
                        ref_group = unique_group_and_only_ozid_samples[0]
                        ref_samples_3 = unique_group_and_only_ozid_samples[1]
                        if ref_group == sample_definitions_group and len(ref_samples_3) > 0:
                            has_ozid = True
                            has_ozid_only = True
                    if not has_ozid:
                        for unique_group_and_ozid_samples in unique_groups_and_ozid_samples:
                            ref_group = unique_group_and_ozid_samples[0]
                            ref_samples_4 = unique_group_and_ozid_samples[1]
                            if ref_group == sample_definitions_group and len(ref_samples_4) > 0:
                                has_ozid = True
                    if has_ozid or has_oznox:
                        unique_groups_and_ozid_oznox_bools += [[sample_definitions_group, has_oznox_only, has_oznox,
                                                                has_ozid_only, has_ozid]]
                    print('')
                    print('For sample group: ' + sample_definitions_group)
                    if has_ozid_only:
                        print('When User RT is not available, the following samples will be queried: ')
                        for ref_sample in ref_samples_3:
                            print(ref_sample)
                    elif has_ozid:
                        print('When User RT is not available, the following samples will be queried: ')
                        for ref_sample in ref_samples_4:
                            print(ref_sample)
                    if has_oznox_only:
                        print('To assign double-bond locations, the following samples will be queried: ')
                        for ref_sample in ref_samples_1:
                            print(ref_sample)
                    elif has_oznox:
                        print('To assign double-bond locations, the following samples will be queried: ')
                        for ref_sample in ref_samples_2:
                            print(ref_sample)
                        print('Warning: assigning double-bond locations by OzID MS1 is unreliable.')
                    elif has_ozid_only:
                        print('To assign double-bond locations, the following samples will be queried: ')
                        for ref_sample in ref_samples_3:
                            print(ref_sample)
                        print('Warning: assigning double-bond locations by OzID MS1 is unreliable.')
                        using_ozid = True
                    else:
                        param_files_pass = False
                        print('Error: double-bond locations cannot be assigned with these definitions.')
            if param_files_pass:
                input_done_2 = False
                while not input_done_2:
                    print('')
                    user_input = input('Proceed? Enter Y for Yes or N for No: ')
                    if user_input == 'Y' or user_input == 'y':
                        input_done_2 = True
                        input_done = True
                    elif user_input == 'N' or user_input == 'n':
                        input_done_2 = True
                    else:
                        print('Error: Invalid user input.  Enter Y or N.')

    script_6_output_frame_index = script_6_output_frame.index
    script_6_output_frame_size = script_6_output_frame_index.size
    print('Performing group-wise sample spectrum alignment ...')
    for i in range(script_6_output_frame_size):
        script_6_output_row = script_6_output_frame.loc[i]
        script_6_output_row_ms_level = int(script_6_output_row['MS Level'])
        if script_6_output_row_ms_level == 2 or using_ozid:

            for group_i in sample_definitions_groups:

                has_oznox_only = False
                has_oznox = False
                has_ozid_only = False

                for unique_group_and_ozid_oznox_bools in unique_groups_and_ozid_oznox_bools:
                    ref_group = unique_group_and_ozid_oznox_bools[0]
                    if ref_group == group_i:
                        has_oznox_only = unique_group_and_ozid_oznox_bools[1]
                        has_oznox = unique_group_and_ozid_oznox_bools[2]
                        has_ozid_only = unique_group_and_ozid_oznox_bools[3]

                main_sample_names = []
                if has_oznox_only:
                    for reference_set in unique_groups_and_only_oznox_samples:
                        unique_group_ref = reference_set[0]
                        if unique_group_ref == group_i:
                            main_sample_names = reference_set[1]
                elif has_oznox:
                    for reference_set in unique_groups_and_oznox_samples:
                        unique_group_ref = reference_set[0]
                        if unique_group_ref == group_i:
                            main_sample_names = reference_set[1]
                elif has_ozid_only:
                    for reference_set in unique_groups_and_only_ozid_samples:
                        unique_group_ref = reference_set[0]
                        if unique_group_ref == group_i:
                            main_sample_names = reference_set[1]
                else:
                    for reference_set in unique_groups_and_ozid_samples:
                        unique_group_ref = reference_set[0]
                        if unique_group_ref == group_i:
                            main_sample_names = reference_set[1]

                spectra = []
                current_spectrum_index = 0
                master_spectrum_index = 0
                highest_counts = 0
                any_nonzero_spectrum = False
                for main_sample_name in main_sample_names:
                    nonzero_spectrum = False
                    spectrum_column_name = 'RT:Counts Spectrum (' + main_sample_name + ')'
                    script_6_output_row_spectrum = (str(script_6_output_row[spectrum_column_name]))
                    if len(script_6_output_row_spectrum) < 5:
                        spectrum_upper = script_6_output_row_spectrum.upper()
                        if len(script_6_output_row_spectrum) == 0 or spectrum_upper == 'NAN':
                            script_6_output_row_spectrum = '0:0'
                        else:
                            nonzero_spectrum = True
                            any_nonzero_spectrum = True
                    else:
                        nonzero_spectrum = True
                        any_nonzero_spectrum = True
                    if nonzero_spectrum:
                        new_spectrum = []
                        spectrum_elements = script_6_output_row_spectrum.split(' ')
                        for spectrum_element in spectrum_elements:
                            element_pieces = spectrum_element.split(':')
                            element_rt = float(element_pieces[0])
                            element_counts = float(element_pieces[1])
                            new_spectrum += [[element_rt, element_counts]]
                            if element_counts > highest_counts:
                                master_spectrum_index = current_spectrum_index
                                highest_counts = element_counts
                        spectra += [new_spectrum]
                        current_spectrum_index += 1

                if any_nonzero_spectrum:

                    master_rt_values = []
                    master_spectrum = spectra[master_spectrum_index]
                    for master_spectrum_element in master_spectrum:
                        master_element_rt = master_spectrum_element[0]
                        master_rt_values += [master_element_rt]

                    averaged_spectra = []
                    for q in range(len(master_rt_values)):
                        master_rt_value = master_rt_values[q]
                        previous_master_rt_value = - 1.0
                        next_master_rt_value = - 1.0
                        has_previous_value = False
                        has_next_value = False
                        if q > 0:
                            previous_master_rt_value = master_rt_values[(q - 1)]
                            has_previous_value = True
                        if q <= (len(master_rt_values) - 2):
                            next_master_rt_value = master_rt_values[(q + 1)]
                            has_next_value = True
                        if has_previous_value and has_next_value:
                            rt_diff_cutoff_1 = abs(master_rt_value - previous_master_rt_value) / 2.0
                            rt_diff_cutoff_2 = abs(master_rt_value - next_master_rt_value) / 2.0
                            if rt_diff_cutoff_1 < rt_diff_cutoff_2:
                                rt_diff_cutoff = rt_diff_cutoff_1
                            else:
                                rt_diff_cutoff = rt_diff_cutoff_2
                        elif has_next_value:
                            rt_diff_cutoff = abs(master_rt_value - next_master_rt_value) / 2.0
                        else:
                            rt_diff_cutoff = abs(master_rt_value - previous_master_rt_value) / 2.0
                        counts_sum = 0.0
                        for spectrum in spectra:
                            best_rt_diff = -1
                            new_counts = 0
                            below_cutoff = False
                            for spectrum_element in spectrum:
                                element_rt = spectrum_element[0]
                                rt_diff = abs(master_rt_value - element_rt)
                                if rt_diff < rt_diff_cutoff:
                                    below_cutoff = True
                                if best_rt_diff == -1:
                                    best_rt_diff = rt_diff
                                    new_counts = spectrum_element[1]
                                elif rt_diff < best_rt_diff:
                                    best_rt_diff = rt_diff
                                    new_counts = spectrum_element[1]
                            if below_cutoff:
                                counts_sum += new_counts
                        counts_averaged = counts_sum / len(main_sample_names)
                        averaged_spectra += [[master_rt_value, counts_averaged]]

                else:

                    averaged_spectra = [[0.0, 0.0]]

                averaged_spectrum_string = ''
                first_element = True
                for averaged_spectra_element in averaged_spectra:
                    if not first_element:
                        averaged_spectrum_string += ' '
                    else:
                        first_element = False
                    element_rt = averaged_spectra_element[0]
                    element_value = averaged_spectra_element[1]
                    averaged_spectrum_string += str(element_rt)
                    averaged_spectrum_string += ':'
                    averaged_spectrum_string += str(element_value)

                new_column_name = 'Averaged Spectrum (' + group_i + ')'
                script_6_output_frame.loc[i, new_column_name] = averaged_spectrum_string
                temp_frame = script_6_output_frame.copy()
                script_6_output_frame = temp_frame.copy()
    print('Group-wise sample spectrum alignment complete.')

    output_columns = ['Parent ID', 'Parent Annotation', 'Class Key', 'Tail Key', 'Parent m/z', 'Double-Bond n-#',
                      'Oz(NOx) Key(s)']
    for sample_definitions_group in sample_definitions_groups:
        group_corrected_quant_column = 'Isomer Quantification Factor (' + sample_definitions_group + ')'
        group_sum_corrected_counts_column = 'Corrected Counts (' + sample_definitions_group + ')'
        output_columns += [group_corrected_quant_column]
        output_columns += [group_sum_corrected_counts_column]
    output_columns += ['Parent RT']
    output_columns += ['All Samples RT of Max Log Score']
    output_columns += ['All Samples Max Log Score']
    for sample_definitions_group in sample_definitions_groups:
        group_max_counts_product_rt_column = 'RT of Max Log Score (' + sample_definitions_group + ')'
        group_max_counts_product_column = 'Max Log Score (' + sample_definitions_group + ')'
        output_columns += [group_max_counts_product_rt_column]
        output_columns += [group_max_counts_product_column]
    for sample_definitions_group in sample_definitions_groups:
        group_counts_product_spectrum_column = 'RT:Log Score Spectrum (' + sample_definitions_group + ')'
        output_columns += [group_counts_product_spectrum_column]
    output_frame = pd.DataFrame(index=[0], columns=output_columns)
    output_index = 0

    parent_counter = 0
    print('')

    # iterating through parent compounds
    for parent_id in parent_ids:

        isomer_indexes = []

        parent_counter += 1

        # searching for double-bond products
        parent_index = None
        product_indexes = []
        for i in range(script_6_output_frame_size):
            script_6_output_row = script_6_output_frame.loc[i]
            script_6_output_row_parent_id = int(script_6_output_row['Parent ID'])
            if script_6_output_row_parent_id == parent_id:
                script_6_output_row_is_parent = str(script_6_output_row['Is Parent'])
                script_6_output_row_is_parent = script_6_output_row_is_parent.upper()
                if script_6_output_row_is_parent == 'TRUE':
                    parent_index = i
                else:
                    script_6_output_row_ms_level = int(script_6_output_row['MS Level'])
                    if script_6_output_row_ms_level == 2 or using_ozid:
                        product_indexes += [i]

        parent_row = script_6_output_frame.loc[parent_index]
        parent_annotation = parent_row['Annotation']
        print('On parent compound ' + str(parent_counter) + ' of ' + str(len(parent_ids)) + ': ' + parent_annotation)
        parent_class = str(parent_row['Class Key'])
        parent_tail_key = str(parent_row['Tail Key'])
        parent_mz = float(parent_row['m/z'])
        parent_rt = str(parent_row['User RT'])
        if len(parent_rt) == 0 or parent_rt.upper() == 'NAN':
            parent_rt = -1
        else:
            parent_rt = float(parent_rt)

        definition_num_tail_chains = None
        search_done = False
        i = -1
        while not search_done:
            i += 1
            class_definitions_row = class_definitions_frame.loc[i]
            class_definitions_row_class = class_definitions_row['Class Key']
            if class_definitions_row_class == parent_class:
                search_done = True
                definition_num_tail_chains = class_definitions_row['Num Tail Chains']

        parent_tail_key_object = TailKey()
        parent_tail_key_object.create(parent_tail_key, definition_num_tail_chains)
        is_molecular = parent_tail_key_object.is_molecular_species
        parent_tail_dbs = parent_tail_key_object.num_doublebonds
        tail_chains = parent_tail_key_object.cleaned_chains

        if parent_tail_key_object.is_p_species and not is_molecular:
            parent_tail_dbs -= 1
        else:
            for tail_chain in tail_chains:
                temp_object = TailKey()
                temp_object.create(tail_chain, 1)
                if temp_object.is_p_species:
                    parent_tail_dbs -= 1

        oznox_keys_indexes = []
        unique_individual_oznox_keys = []
        for product_index in product_indexes:
            script_6_output_row = script_6_output_frame.loc[product_index]
            script_6_output_row_oznox_key = script_6_output_row['Oz(NOx) Key']
            space_index = script_6_output_row_oznox_key.find(' ')
            if space_index > 0:
                oznox_key_parts = script_6_output_row_oznox_key.split(' ')
                for oznox_key_part in oznox_key_parts:
                    oznox_keys_indexes += [[oznox_key_part, product_index]]
                    new_oznox_key = True
                    for unique_individual_oznox_key in unique_individual_oznox_keys:
                        if unique_individual_oznox_key == oznox_key_part:
                            new_oznox_key = False
                    if new_oznox_key:
                        unique_individual_oznox_keys += [oznox_key_part]
            else:
                oznox_keys_indexes += [[script_6_output_row_oznox_key, product_index]]
                new_oznox_key = True
                for unique_individual_oznox_key in unique_individual_oznox_keys:
                    if unique_individual_oznox_key == script_6_output_row_oznox_key:
                        new_oznox_key = False
                if new_oznox_key:
                    unique_individual_oznox_keys += [script_6_output_row_oznox_key]

        # continue with this parent compound if there are product annotations
        if len(unique_individual_oznox_keys) > 0:

            max_n_nums_per_tail = []
            if is_molecular:
                for tail_chain in tail_chains:
                    tail_chain_object = TailKey()
                    tail_chain_object.create(tail_chain, 1)
                    tail_length = tail_chain_object.num_carbons
                    if tail_chain_object.is_t_species or tail_chain_object.is_d_species:
                        tail_length -= 4
                    elif tail_chain_object.is_p_species:
                        tail_length -= 4
                    elif tail_chain_object.is_o_species:
                        tail_length -= 3
                    else:
                        tail_length -= 2
                    max_n_nums_per_tail += [tail_length]
            else:
                tail_length = parent_tail_key_object.num_carbons
                tail_length -= (3 * (definition_num_tail_chains - 1))
                q = 0
                while q < definition_num_tail_chains:
                    q += 1
                    max_n_nums_per_tail += [tail_length]

            possible_tail_db_count_sets = []
            if parent_tail_key_object.num_doublebonds > 0:
                if is_molecular:
                    new_db_counts_set = []
                    for tail_chain in tail_chains:
                        tail_chain_object = TailKey()
                        tail_chain_object.create(tail_chain, 1)
                        is_p_species = tail_chain_object.is_p_species
                        tail_chain_dbs = tail_chain_object.num_doublebonds
                        if is_p_species:
                            tail_chain_dbs -= 1
                        if tail_chain_dbs < 0:
                            tail_chain_dbs = 0
                        new_db_counts_set += [tail_chain_dbs]
                    possible_tail_db_count_sets += [new_db_counts_set]
                else:
                    db_count_combos = []
                    i = -1
                    while i < parent_tail_dbs:
                        i += 1
                        db_count_combos += [[i]]
                    db_count_combos = recursive_db_combo_builder(db_count_combos, definition_num_tail_chains,
                                                                 parent_tail_dbs)
                    for db_count_combo in db_count_combos:
                        possible_tail_db_count_sets += [db_count_combo]

            temp_array = []
            for possible_tail_db_count_set in possible_tail_db_count_sets:
                max_db_count = -1
                for set_element in possible_tail_db_count_set:
                    if set_element > max_db_count:
                        max_db_count = set_element
                found_match = False
                for unique_individual_oznox_key in unique_individual_oznox_keys:
                    key_values_i = read_ozid_key(unique_individual_oznox_key)
                    db_value_i = key_values_i[1] + 1
                    if db_value_i == max_db_count:
                        found_match = True
                if found_match:
                    temp_array += [possible_tail_db_count_set]
            possible_tail_db_count_sets = temp_array

            master_oznox_key_combos = []
            for possible_tail_db_count_set in possible_tail_db_count_sets:
                good_set = True
                oznox_key_combos = []
                k = 0
                while k < definition_num_tail_chains:
                    if is_molecular:
                        tail_chain_i = tail_chains[k]
                    else:
                        tail_chain_i = parent_tail_key
                    max_n_num_per_tail = max_n_nums_per_tail[k]
                    possible_tail_db_count = possible_tail_db_count_set[k]
                    k += 1
                    new_tail_combos = []
                    found_base_db_count = False
                    for unique_individual_oznox_key in unique_individual_oznox_keys:
                        oznox_key_values = read_ozid_key(unique_individual_oznox_key)
                        neutral_loss_dbs = oznox_key_values[1]
                        n_pos = oznox_key_values[0]
                        if neutral_loss_dbs == (possible_tail_db_count - 1) and n_pos <= max_n_num_per_tail:
                            found_base_db_count = True
                            new_tail_combo = [unique_individual_oznox_key]
                            new_tail_combos += [new_tail_combo]
                    if found_base_db_count:
                        new_tail_combos = recursive_oznox_combo_builder(tail_chain_i, new_tail_combos,
                                                                        unique_individual_oznox_keys, is_molecular,
                                                                        definition_num_tail_chains)
                        if len(oznox_key_combos) == 0 and len(new_tail_combos) > 0:
                            for new_tail_combo in new_tail_combos:
                                oznox_key_combos += [new_tail_combo]
                        else:
                            if len(new_tail_combos) > 0:
                                temp_array = []
                                for oznox_key_combo in oznox_key_combos:
                                    new_item = [oznox_key_combo]
                                    for new_tail_combo in new_tail_combos:
                                        new_item_i = copy.deepcopy(new_item)
                                        new_item_i += [new_tail_combo]
                                        temp_array += [new_item_i]
                                oznox_key_combos = temp_array
                            else:
                                good_set = False
                if good_set:
                    for oznox_key_combo in oznox_key_combos:
                        master_oznox_key_combos += [oznox_key_combo]

            # continue with this parent compound if there is a product set that meets the basic requirements
            unique_oznox_key_sets = []
            if len(master_oznox_key_combos) > 0:
                for master_oznox_key_combo in master_oznox_key_combos:
                    new_oznox_key_set = []
                    for master_oznox_key_combo_element in master_oznox_key_combo:
                        element_copy = copy.deepcopy(master_oznox_key_combo_element)
                        master_oznox_key_combo_element_string = str(element_copy)
                        comma_index = master_oznox_key_combo_element_string.find(', ')
                        if comma_index > 0:
                            for oznox_key_element in master_oznox_key_combo_element:
                                new_oznox_key_set += [oznox_key_element]
                        else:
                            new_oznox_key_set += [master_oznox_key_combo_element]

                    cleaned_set = []
                    for new_oznox_key_set_element in new_oznox_key_set:
                        element_string = str(new_oznox_key_set_element)
                        bracket_index = element_string.find('[')
                        if bracket_index >= 0:
                            for element_i in new_oznox_key_set_element:
                                element_i_string = str(element_i)
                                bracket_index_i = element_i_string.find('[')
                                if bracket_index_i >= 0:
                                    for element_j in element_i:
                                        cleaned_set += [element_j]
                                else:
                                    cleaned_set += [element_i]
                        else:
                            cleaned_set += [new_oznox_key_set_element]

                    sorted_new_oznox_key_set = sort_oznox_keys(cleaned_set)
                    novel_combo = True
                    for unique_oznox_key_set in unique_oznox_key_sets:
                        if sorted_new_oznox_key_set == unique_oznox_key_set:
                            novel_combo = False
                    if novel_combo:
                        unique_oznox_key_sets += [sorted_new_oznox_key_set]

            for unique_oznox_key_set in unique_oznox_key_sets:

                all_groups_max_product_rt = 0
                all_groups_max_product = 0

                position_combos = get_db_pos_combos(parent_tail_key, definition_num_tail_chains,
                                                    unique_oznox_key_set)
                good_position_combos = []
                for position_combo in position_combos:
                    good_combo = True
                    for combo_tail in position_combo:
                        for combo_tail_element in combo_tail:
                            if combo_tail_element == -1:
                                good_combo = False
                    if good_combo:
                        tail_db_counts = []
                        for position_combo_element in position_combo:
                            db_counter = len(position_combo_element)
                            tail_db_counts += [db_counter]
                        if check_db_annotation(tail_db_counts, definition_num_tail_chains, unique_oznox_key_set):
                            good_position_combos += [position_combo]

                # continue with this isomer of the parent compound if it is a valid product combination
                has_combo_counts = False
                groups_and_rts_counts_spectrum = {}
                if len(good_position_combos) > 0:

                    # iterating through sample groups
                    for group_i in sample_definitions_groups:

                        has_oznox = False

                        for unique_group_and_ozid_oznox_bools in unique_groups_and_ozid_oznox_bools:
                            ref_group = unique_group_and_ozid_oznox_bools[0]
                            if ref_group == group_i:
                                has_oznox = unique_group_and_ozid_oznox_bools[2]

                        unique_keys_i = []
                        key_reps = {}
                        keys_fa_losses = {}
                        keys_and_summed_counts = {}
                        for unique_oznox_key_set_element in unique_oznox_key_set:
                            keys_and_summed_counts[unique_oznox_key_set_element] = 0.0
                            keys_fa_losses[unique_oznox_key_set_element] = '-1'
                            new_element = True
                            for unique_key_i in unique_keys_i:
                                if unique_key_i == unique_oznox_key_set_element:
                                    new_element = False
                            if new_element:
                                key_reps[unique_oznox_key_set_element] = 1
                                unique_keys_i += [unique_oznox_key_set_element]
                            else:
                                old_reps = key_reps[unique_oznox_key_set_element]
                                new_reps = old_reps + 1
                                key_reps[unique_oznox_key_set_element] = new_reps

                        keys_and_averaged_spectra = []
                        for unique_oznox_key_set_element in unique_keys_i:
                            for oznox_key_index_set in oznox_keys_indexes:
                                oznox_key_ref = oznox_key_index_set[0]
                                if oznox_key_ref == unique_oznox_key_set_element:
                                    oznox_index_ref = oznox_key_index_set[1]
                                    script_6_output_row = script_6_output_frame.loc[oznox_index_ref]
                                    script_6_output_row_ms_level = int(script_6_output_row['MS Level'])
                                    if script_6_output_row_ms_level == 2 or (not has_oznox):
                                        spectrum_column_name = 'Averaged Spectrum (' + group_i + ')'
                                        script_6_output_row_spectrum = (str(script_6_output_row[spectrum_column_name]))
                                        if len(script_6_output_row_spectrum) < 5:
                                            spectrum_upper = script_6_output_row_spectrum.upper()
                                            if len(script_6_output_row_spectrum) == 0 or spectrum_upper == 'NAN':
                                                spectrum_array = [[0.0, 0.0]]
                                            elif script_6_output_row_spectrum == '0:0':
                                                spectrum_array = [[0.0, 0.0]]
                                            else:
                                                spectrum_array = []
                                                spectrum_elements = script_6_output_row_spectrum.split(' ')
                                                for spectrum_element in spectrum_elements:
                                                    element_pieces = spectrum_element.split(':')
                                                    element_rt = float(element_pieces[0])
                                                    element_value = float(element_pieces[1])
                                                    spectrum_array += [[element_rt, element_value]]
                                        else:
                                            spectrum_array = []
                                            spectrum_elements = script_6_output_row_spectrum.split(' ')
                                            for spectrum_element in spectrum_elements:
                                                element_pieces = spectrum_element.split(':')
                                                element_rt = float(element_pieces[0])
                                                element_value = float(element_pieces[1])
                                                spectrum_array += [[element_rt, element_value]]

                                        row_fa_loss = str(script_6_output_row['Scheme 2 FA Loss'])
                                        row_fa_loss = row_fa_loss.upper()
                                        if len(row_fa_loss) == 0 or row_fa_loss == 'NAN':
                                            row_fa_loss = 'NAN'
                                        if row_fa_loss != 'NAN':
                                            cleaning_object = TailKey()
                                            cleaning_object.create(row_fa_loss, 1)
                                            simplified_chains = cleaning_object.simplified_chains
                                            row_fa_loss = simplified_chains[0]
                                        keys_and_averaged_spectra += [[unique_oznox_key_set_element, spectrum_array,
                                                                       row_fa_loss]]

                        has_duplicate_losable_tails = False
                        has_unique_losable_tails = False
                        unique_losable_tails = []
                        all_losable_tails = []
                        for p in range(len(tail_chains)):
                            cleaned_tail = tail_chains[p]
                            unique_tail_object = TailKey()
                            unique_tail_object.create(cleaned_tail, 1)
                            is_fa = True
                            if unique_tail_object.is_p_species:
                                is_fa = False
                            elif unique_tail_object.is_o_species:
                                is_fa = False
                            elif unique_tail_object.is_q_species:
                                is_fa = False
                            elif unique_tail_object.is_d_species:
                                is_fa = False
                            elif unique_tail_object.is_t_species:
                                is_fa = False
                            if is_fa:
                                another_chain_has_dbs = False
                                for t in range(len(tail_chains)):
                                    if t != p:
                                        cleaned_tail_t = tail_chains[t]
                                        if cleaned_tail_t != cleaned_tail:
                                            tail_object_t = TailKey()
                                            tail_object_t.create(cleaned_tail_t, 1)
                                            db_count_t = tail_object_t.num_doublebonds
                                            if tail_object_t.is_p_species:
                                                db_count_t -= 1
                                            if db_count_t >= 1:
                                                another_chain_has_dbs = True
                                if another_chain_has_dbs:
                                    all_losable_tails += [cleaned_tail]
                                    new_unique_tail = True
                                    for unique_tail in unique_losable_tails:
                                        if cleaned_tail == unique_tail:
                                            new_unique_tail = False
                                    if new_unique_tail:
                                        unique_losable_tails += [cleaned_tail]
                                    else:
                                        has_duplicate_losable_tails = True

                        unique_losable_tails_counts = {}
                        for unique_tail in unique_losable_tails:
                            match_counter = 0
                            for cleaned_tail in tail_chains:
                                if cleaned_tail == unique_tail:
                                    match_counter += 1
                            if match_counter == 1:
                                has_unique_losable_tails = True
                            cleaning_object = TailKey()
                            cleaning_object.create(unique_tail, 1)
                            simplified_chains = cleaning_object.simplified_chains
                            simplified_chain = simplified_chains[0]
                            unique_losable_tails_counts[simplified_chain] = match_counter

                        averaged_spectra_master_index = -1
                        averaged_spectra_max_counts = 0
                        for d in range(len(keys_and_averaged_spectra)):
                            key_and_averaged_spectrum = keys_and_averaged_spectra[d]
                            averaged_spectrum_key = key_and_averaged_spectrum[0]
                            averaged_spectrum = key_and_averaged_spectrum[1]
                            averaged_spectrum_fa_loss = key_and_averaged_spectrum[2]
                            old_fa_losses = keys_fa_losses[averaged_spectrum_key]
                            if old_fa_losses == '-1':
                                keys_fa_losses[averaged_spectrum_key] = [averaged_spectrum_fa_loss]
                            else:
                                new_fa_losses = old_fa_losses
                                new_fa_losses += [averaged_spectrum_fa_loss]
                                keys_fa_losses[averaged_spectrum_key] = new_fa_losses
                            for spectrum_element in averaged_spectrum:
                                spectrum_element_counts = spectrum_element[1]
                                if spectrum_element_counts > averaged_spectra_max_counts:
                                    averaged_spectra_max_counts = spectrum_element_counts
                                    averaged_spectra_master_index = d

                        master_rt_values = []
                        key_and_averaged_spectrum_i = keys_and_averaged_spectra[averaged_spectra_master_index]
                        spectrum_i = key_and_averaged_spectrum_i[1]
                        for spectrum_element in spectrum_i:
                            spectrum_element_rt = spectrum_element[0]
                            master_rt_values += [spectrum_element_rt]

                        product_spectrum = []
                        group_i_rt = -1
                        group_i_max_prod = 0
                        best_intensity_set = []

                        for c in range(len(master_rt_values)):
                            if c > 0:
                                previous_master_rt_value = master_rt_values[(c - 1)]
                                has_previous_rt = True
                            else:
                                previous_master_rt_value = -1
                                has_previous_rt = False
                            current_master_rt_value = master_rt_values[c]
                            if c <= (len(master_rt_values) - 2):
                                next_master_rt_value = master_rt_values[(c + 1)]
                                has_next_rt = True
                            else:
                                has_next_rt = False
                                next_master_rt_value = -1
                            if has_next_rt and has_previous_rt:
                                rt_diff_cutoff_1 = abs(current_master_rt_value - previous_master_rt_value) / 2.0
                                rt_diff_cutoff_2 = abs(current_master_rt_value - next_master_rt_value) / 2.0
                                if rt_diff_cutoff_1 < rt_diff_cutoff_2:
                                    master_rt_cutoff = rt_diff_cutoff_1
                                else:
                                    master_rt_cutoff = rt_diff_cutoff_2
                            elif has_next_rt:
                                master_rt_cutoff = abs(next_master_rt_value - current_master_rt_value) / 2.0
                            else:
                                master_rt_cutoff = abs(current_master_rt_value - previous_master_rt_value) / 2.0

                            rt_counts_product = 0.0
                            rt_counts_sum = 0.0
                            keys_with_nonzero = []
                            keys_with_zero = []
                            temp_key_intensity_set = []

                            for key_and_averaged_spectrum_i in keys_and_averaged_spectra:
                                key_i = key_and_averaged_spectrum_i[0]
                                key_reps_i = key_reps[key_i]
                                key_fa_losses_i = keys_fa_losses[key_i]
                                key_fa_losses_i_num = 0
                                has_nan = False
                                has_not_nan = False
                                for key_fa_losses_i_element in key_fa_losses_i:
                                    if key_fa_losses_i_element != 'NAN':
                                        key_fa_losses_i_num += 1
                                        has_not_nan = True
                                    else:
                                        has_nan = True
                                spectrum_i = key_and_averaged_spectrum_i[1]
                                fa_loss_i = key_and_averaged_spectrum_i[2]
                                if fa_loss_i != 'NAN':
                                    simplification_object = TailKey()
                                    simplification_object.create(fa_loss_i, 1)
                                    simplified_chains = simplification_object.simplified_chains
                                    fa_loss_i = simplified_chains[0]
                                if fa_loss_i != 'NAN' and has_unique_losable_tails and has_duplicate_losable_tails:
                                    losable_count_i = unique_losable_tails_counts[fa_loss_i]
                                    if key_fa_losses_i_num >= losable_count_i:
                                        losable_count_i = 1.0
                                else:
                                    losable_count_i = 1.0
                                f = -1
                                found_rt = False
                                finished_scanning = False
                                closest_f_index = 0
                                closest_index_rt_diff = -1
                                while f < (len(spectrum_i) - 1) and not finished_scanning:
                                    f += 1
                                    spectrum_element = spectrum_i[f]
                                    spectrum_element_rt = spectrum_element[0]
                                    rt_diff_i = abs(current_master_rt_value - spectrum_element_rt)
                                    if closest_index_rt_diff == -1 or rt_diff_i < closest_index_rt_diff:
                                        closest_f_index = f
                                        closest_index_rt_diff = rt_diff_i
                                    if rt_diff_i <= master_rt_cutoff:
                                        found_rt = True
                                        closest_f_index = f
                                    if found_rt and rt_diff_i > closest_index_rt_diff:
                                        finished_scanning = True
                                if found_rt:
                                    spectrum_element = spectrum_i[closest_f_index]
                                    spectrum_element_counts = spectrum_element[1]
                                    rt_counts_sum += spectrum_element_counts
                                else:
                                    spectrum_element_counts = 0.0
                                temp_key_intensity_set += [[key_i, spectrum_element_counts]]
                                if spectrum_element_counts > 0:
                                    keys_with_nonzero += [key_i]
                                    final_exponent = 1.0
                                    if fa_loss_i != 'NAN':
                                        spectrum_element_counts = spectrum_element_counts / losable_count_i
                                        final_exponent += (losable_count_i - 1.0)
                                    spectrum_element_counts = spectrum_element_counts / key_reps_i
                                    final_exponent += (key_reps_i - 1.0)
                                    spectrum_element_counts = log10(spectrum_element_counts)
                                    if has_nan and has_not_nan:
                                        final_exponent += 0.0
                                    spectrum_element_counts = spectrum_element_counts * final_exponent
                                else:
                                    keys_with_zero += [key_i]
                                rt_counts_product = rt_counts_product + spectrum_element_counts
                            has_zero = False
                            for key_z in keys_with_zero:
                                found_replacement = False
                                for key_nz in keys_with_nonzero:
                                    if key_nz == key_z:
                                        found_replacement = True
                                if not found_replacement:
                                    has_zero = True
                            if has_zero:
                                rt_counts_product = 0.0
                            product_spectrum += [[current_master_rt_value, rt_counts_product]]
                            if rt_counts_product > group_i_max_prod:
                                group_i_max_prod = rt_counts_product
                                group_i_rt = current_master_rt_value
                                best_intensity_set = temp_key_intensity_set

                        for set_element_i in best_intensity_set:
                            key_i = set_element_i[0]
                            value_i = set_element_i[1]
                            old_summed_value = keys_and_summed_counts[key_i]
                            new_summed_value = old_summed_value + value_i
                            keys_and_summed_counts[key_i] = new_summed_value

                        for key_i in unique_keys_i:
                            old_summed_value = keys_and_summed_counts[key_i]
                            key_reps_i = key_reps[key_i]
                            new_summed_value = old_summed_value / key_reps_i
                            keys_and_summed_counts[key_i] = new_summed_value

                        lowest_summed_value = -1
                        for key_i in unique_keys_i:
                            summed_value = keys_and_summed_counts[key_i]
                            if lowest_summed_value == -1 or summed_value < lowest_summed_value:
                                lowest_summed_value = summed_value

                        group_i_sum_prod_counts = lowest_summed_value * parent_tail_dbs

                        group_i_prod_spectrum = ''
                        first_element = True
                        for product_spectrum_element in product_spectrum:
                            if not first_element:
                                group_i_prod_spectrum += ' '
                            else:
                                first_element = False
                            product_spectrum_element_rt = product_spectrum_element[0]
                            product_spectrum_element_product = product_spectrum_element[1]
                            group_i_prod_spectrum += (str(product_spectrum_element_rt) + ':'
                                                      + str(product_spectrum_element_product))

                        if group_i_max_prod > all_groups_max_product:
                            all_groups_max_product = group_i_max_prod
                            all_groups_max_product_rt = group_i_rt

                        if group_i_max_prod > 0:
                            has_combo_counts = True
                        else:
                            group_i_rt = ''
                            group_i_max_prod = ''
                            group_i_prod_spectrum = ''
                        groups_and_rts_counts_spectrum = {group_i: [group_i_rt, group_i_max_prod, group_i_prod_spectrum,
                                                                    group_i_sum_prod_counts]}

                if has_combo_counts:

                    n_string = '('
                    first_combo = True
                    for position_combo in good_position_combos:
                        if not first_combo:
                            n_string += ') | ('
                        first_combo = False
                        first_tail = True
                        for combo_tail in position_combo:
                            if len(combo_tail) > 0:
                                if not first_tail:
                                    n_string += '_'
                                n_string += 'n-'
                                first_tail = False
                                first_pos = True
                                for combo_tail_pos in combo_tail:
                                    if not first_pos:
                                        n_string += ','
                                    first_pos = False
                                    n_string += str(combo_tail_pos)
                    n_string += ')'

                    oznox_keys_string = ''
                    first_key = True
                    for unique_oznox_key in unique_oznox_key_set:
                        if not first_key:
                            oznox_keys_string += ' '
                        first_key = False
                        oznox_keys_string += unique_oznox_key

                    isomer_indexes += [output_index]

                    if output_index == 0:
                        output_frame.loc[output_index, 'Parent ID'] = parent_id
                        output_frame.loc[output_index, 'Parent Annotation'] = parent_annotation
                        output_frame.loc[output_index, 'Class Key'] = parent_class
                        output_frame.loc[output_index, 'Tail Key'] = parent_tail_key
                        output_frame.loc[output_index, 'Parent m/z'] = parent_mz
                        output_frame.loc[output_index, 'Double-Bond n-#'] = n_string
                        output_frame.loc[output_index, 'Oz(NOx) Key(s)'] = oznox_keys_string
                        output_frame.loc[output_index, 'Parent RT'] = parent_rt
                        output_frame.loc[output_index, 'All Samples RT of Max Log Score'] = (
                            all_groups_max_product_rt)
                        output_frame.loc[output_index, 'All Samples Max Log Score'] = all_groups_max_product
                        for group_i in sample_definitions_groups:
                            group_i_rt_column = 'RT of Max Log Score (' + group_i + ')'
                            group_i_counts_column = 'Max Log Score (' + group_i + ')'
                            group_i_spectrum_column = 'RT:Log Score Spectrum (' + group_i + ')'
                            group_i_values = groups_and_rts_counts_spectrum[group_i]
                            group_i_rt = group_i_values[0]
                            group_i_counts = group_i_values[1]
                            group_i_spectrum = group_i_values[2]
                            output_frame.loc[output_index, group_i_rt_column] = group_i_rt
                            output_frame.loc[output_index, group_i_counts_column] = group_i_counts
                            output_frame.loc[output_index, group_i_spectrum_column] = group_i_spectrum
                        temp_frame = output_frame.copy()
                        output_frame = temp_frame.copy()
                        output_index += 1
                    else:
                        data = {'Parent ID': parent_id, 'Parent Annotation': parent_annotation,
                                'Class Key': parent_class, 'Tail Key': parent_tail_key, 'Parent m/z': parent_mz,
                                'Double-Bond n-#': n_string, 'Oz(NOx) Key(s)': oznox_keys_string,
                                'Parent RT': parent_rt,
                                'All Samples RT of Max Log Score': all_groups_max_product_rt,
                                'All Samples Max Log Score': all_groups_max_product}
                        for group_i in sample_definitions_groups:
                            group_i_rt_column = 'RT of Max Log Score (' + group_i + ')'
                            group_i_counts_column = 'Max Log Score (' + group_i + ')'
                            group_i_spectrum_column = 'RT:Log Score Spectrum (' + group_i + ')'
                            group_i_values = groups_and_rts_counts_spectrum[group_i]
                            group_i_rt = group_i_values[0]
                            group_i_counts = group_i_values[1]
                            group_i_spectrum = group_i_values[2]
                            data[group_i_rt_column] = group_i_rt
                            data[group_i_counts_column] = group_i_counts
                            data[group_i_spectrum_column] = group_i_spectrum
                        data_frame = pd.DataFrame(data=data, index=[0])
                        temp_frame = pd.concat([output_frame, data_frame], ignore_index=True)
                        output_frame = temp_frame.copy()
                        output_index += 1

        # for isomer relative quantification, iterating over sample groups
        for group_i in sample_definitions_groups:

            has_unsolvable_isomers = False
            group_corrected_quant_column = 'Isomer Quantification Factor (' + group_i + ')'
            group_sum_corrected_counts_column = 'Corrected Counts (' + group_i + ')'

            # proceed variable used in iterative assessment of whether relative quantification is possible
            if len(isomer_indexes) == 0:
                proceed = False
            else:
                proceed = True

            # gathering double-bond product species spectra
            oznox_keys_and_averaged_spectra = []
            if proceed:
                for oznox_key_index in oznox_keys_indexes:
                    oznox_key_i = oznox_key_index[0]
                    script_6_index = oznox_key_index[1]
                    script_6_row = script_6_output_frame.loc[script_6_index]
                    averaged_spectrum_column = 'Averaged Spectrum (' + group_i + ')'
                    script_6_spectrum = str(script_6_row[averaged_spectrum_column])
                    script_6_spectrum_array = convert_spectrum_string_to_array(script_6_spectrum)
                    if script_6_spectrum_array != [[0.0, 0.0]]:
                        oznox_keys_and_averaged_spectra += [[oznox_key_i, script_6_spectrum_array]]

            if len(oznox_keys_and_averaged_spectra) == 0:
                proceed = False

            # gathering isomer product spectra and OzNOx keys in reference to isomer indexes
            indexes_key_sets_product_spectra = []
            if proceed:
                for isomer_index in isomer_indexes:
                    output_frame_row = output_frame.loc[isomer_index]
                    key_set_i = str(output_frame_row['Oz(NOx) Key(s)'])
                    space_index = key_set_i.find(' ')
                    if space_index > 0:
                        temp_array = []
                        key_set_i_elements = key_set_i.split(' ')
                        for key_set_i_element in key_set_i_elements:
                            temp_array += [key_set_i_element]
                        key_set_i = temp_array
                    else:
                        temp_array = [key_set_i]
                        key_set_i = temp_array
                    column_name_i = 'RT:Log Score Spectrum (' + group_i + ')'
                    product_spectrum_i = str(output_frame_row[column_name_i])
                    if (len(product_spectrum_i) > 0 and product_spectrum_i != '0:0'
                            and product_spectrum_i.upper() != 'NAN'):
                        product_spectrum_matrix_i = []
                        spectrum_elements_i = product_spectrum_i.split(' ')
                        nonzero_spectrum = False
                        for spectrum_element_i in spectrum_elements_i:
                            spectrum_element_pieces = spectrum_element_i.split(':')
                            spectrum_element_i_rt = float(spectrum_element_pieces[0])
                            spectrum_element_i_value = float(spectrum_element_pieces[1])
                            if spectrum_element_i_value > 0.0:
                                nonzero_spectrum = True
                            product_spectrum_matrix_i += [[spectrum_element_i_rt, spectrum_element_i_value]]
                        if nonzero_spectrum:
                            indexes_key_sets_product_spectra += [[isomer_index, key_set_i, product_spectrum_matrix_i]]

            if len(indexes_key_sets_product_spectra) == 0:
                proceed = False

            # aligning isomer product spectra
            master_rt_values = []
            final_isomer_indexes = []
            isomer_index_dict_to_keys = {}
            isomer_index_dict_to_prod_spectra = {}
            if proceed:

                # determining which product spectrum ought to be the basis for alignment
                master_index = 0
                greatest_product = 0
                for i in range(len(indexes_key_sets_product_spectra)):
                    index_key_set_product_spectrum = indexes_key_sets_product_spectra[i]
                    product_spectrum_i = index_key_set_product_spectrum[2]
                    for spectrum_element in product_spectrum_i:
                        spectrum_element_value = spectrum_element[1]
                        if spectrum_element_value > greatest_product:
                            master_index = i
                            greatest_product = spectrum_element_value

                # setting target rt_values from master product spectrum
                master_spectrum = indexes_key_sets_product_spectra[master_index][2]
                for master_element in master_spectrum:
                    master_rt_value = master_element[0]
                    master_rt_values += [master_rt_value]

                # creating aligned product spectra
                for i in range(len(indexes_key_sets_product_spectra)):
                    index_key_set_product_spectrum = indexes_key_sets_product_spectra[i]
                    isomer_index_i = index_key_set_product_spectrum[0]
                    key_set_i = index_key_set_product_spectrum[1]
                    product_spectrum_i = index_key_set_product_spectrum[2]
                    if i == master_index:
                        final_isomer_indexes += [isomer_index_i]
                        isomer_index_dict_to_keys[isomer_index_i] = key_set_i
                        isomer_index_dict_to_prod_spectra[isomer_index_i] = product_spectrum_i
                    else:
                        isomer_index_i = index_key_set_product_spectrum[0]
                        key_set_i = index_key_set_product_spectrum[1]
                        product_spectrum_i = index_key_set_product_spectrum[2]
                        aligned_spectrum = []

                        # iterating through master RT values
                        all_zero = True
                        for c in range(len(master_rt_values)):
                            if c > 0:
                                previous_master_rt_value = master_rt_values[(c - 1)]
                                has_previous_rt = True
                            else:
                                previous_master_rt_value = -1
                                has_previous_rt = False
                            current_master_rt_value = master_rt_values[c]
                            if c <= (len(master_rt_values) - 2):
                                next_master_rt_value = master_rt_values[(c + 1)]
                                has_next_rt = True
                            else:
                                has_next_rt = False
                                next_master_rt_value = -1
                            if has_next_rt and has_previous_rt:
                                rt_diff_cutoff_1 = abs(current_master_rt_value - previous_master_rt_value) / 2.0
                                rt_diff_cutoff_2 = abs(current_master_rt_value - next_master_rt_value) / 2.0
                                if rt_diff_cutoff_1 < rt_diff_cutoff_2:
                                    master_rt_cutoff = rt_diff_cutoff_1
                                else:
                                    master_rt_cutoff = rt_diff_cutoff_2
                            elif has_next_rt:
                                master_rt_cutoff = abs(next_master_rt_value - current_master_rt_value) / 2.0
                            else:
                                master_rt_cutoff = abs(current_master_rt_value - previous_master_rt_value) / 2.0

                            # searching through product_spectrum_i
                            f = -1
                            found_rt = False
                            finished_scanning = False
                            closest_f_index = 0
                            closest_index_rt_diff = -1
                            while f < (len(product_spectrum_i) - 1) and not finished_scanning:
                                f += 1
                                spectrum_element = product_spectrum_i[f]
                                spectrum_element_rt = spectrum_element[0]
                                rt_diff_i = abs(current_master_rt_value - spectrum_element_rt)
                                if closest_index_rt_diff == -1 or rt_diff_i < closest_index_rt_diff:
                                    closest_f_index = f
                                    closest_index_rt_diff = rt_diff_i
                                if rt_diff_i <= master_rt_cutoff:
                                    found_rt = True
                                    closest_f_index = f
                                if found_rt and rt_diff_i > closest_index_rt_diff:
                                    finished_scanning = True
                            if found_rt:
                                spectrum_element = product_spectrum_i[closest_f_index]
                                spectrum_element_counts = spectrum_element[1]
                            else:
                                spectrum_element_counts = 0.0
                            if spectrum_element_counts > 0.0:
                                all_zero = False
                            aligned_spectrum += [[current_master_rt_value, spectrum_element_counts]]
                        if not all_zero:
                            final_isomer_indexes += [isomer_index_i]
                            isomer_index_dict_to_keys[isomer_index_i] = key_set_i
                            isomer_index_dict_to_prod_spectra[isomer_index_i] = aligned_spectrum

            # aligning double-bond product spectra
            # duplicated keys' spectra are summed
            # a dictionary collection of finished keys' spectra is created
            oznox_keys_and_aligned_spectra = []
            unique_keys = []
            keys_and_finished_spectra = {}
            if proceed:
                for oznox_key_and_averaged_spectrum in oznox_keys_and_averaged_spectra:
                    oznox_key_i = oznox_key_and_averaged_spectrum[0]
                    averaged_spectrum_i = oznox_key_and_averaged_spectrum[1]
                    aligned_spectrum_i = align_spectrum_to_rt_values(averaged_spectrum_i, master_rt_values)

                    if aligned_spectrum_i != [[0.0, 0.0]]:
                        oznox_keys_and_aligned_spectra += [[oznox_key_i, aligned_spectrum_i]]

                        # determining if there are duplicate keys, which can occur with Scheme 2 fragmentation
                        # duplicate key spectra will need to be summed
                        new_unique_key = True
                        for unique_key in unique_keys:
                            if unique_key == oznox_key_i:
                                new_unique_key = False
                        if new_unique_key:
                            unique_keys += [oznox_key_i]
                            keys_and_finished_spectra[oznox_key_i] = aligned_spectrum_i
                        else:
                            old_spectrum = keys_and_finished_spectra[oznox_key_i]
                            new_spectrum = []
                            for j in range(len(old_spectrum)):
                                old_spectrum_element = old_spectrum[j]
                                old_spectrum_rt = old_spectrum_element[0]
                                old_spectrum_value = old_spectrum_element[1]
                                new_spectrum_element = aligned_spectrum_i[j]
                                new_spectrum_value = new_spectrum_element[1]
                                summed_spectrum_value = old_spectrum_value + new_spectrum_value
                                new_spectrum += [[old_spectrum_rt, summed_spectrum_value]]
                            keys_and_finished_spectra[oznox_key_i] = new_spectrum

            # trimming aligned product spectra to eliminate unsolvable scans if they have < 1% of total counts
            # also determining whether this isomer set can be quantified with current algorithm(s)
            nonzero_scan_indexes = []
            scan_index_dict_to_isomer_indexes = {}
            solvable_scan_indexes = []
            total_solvable_counts = 0.0
            unsolvable_scan_indexes = []
            total_unsolvable_counts = 0.0
            scan_index_dict_to_solvable_sets = {}
            scan_index_dict_to_solvable_isomer_indexes = {}
            if proceed:

                # determining what scan indexes are nonzero and what isomer indexes are connected
                for i in range(len(master_rt_values)):
                    for isomer_index_j in final_isomer_indexes:
                        prod_spectrum_j = isomer_index_dict_to_prod_spectra[isomer_index_j]
                        spectrum_element = prod_spectrum_j[i]
                        spectrum_element_value = spectrum_element[1]
                        if spectrum_element_value > 0:
                            new_nonzero_index = True
                            for old_nonzero_index in nonzero_scan_indexes:
                                if old_nonzero_index == i:
                                    new_nonzero_index = False
                            if new_nonzero_index:
                                nonzero_scan_indexes += [i]
                                scan_index_dict_to_isomer_indexes[i] = [isomer_index_j]
                            else:
                                old_isomer_indexes = scan_index_dict_to_isomer_indexes[i]
                                old_isomer_indexes += [isomer_index_j]
                                scan_index_dict_to_isomer_indexes[i] = old_isomer_indexes

                # iterating through nonzero scan indexes to determine if they are solvable or unsolvable within 1%
                # the 1% cutoff is based on the logarithmic product scores at this spectrum index
                for nonzero_scan_index in nonzero_scan_indexes:

                    ref_isomer_indexes = scan_index_dict_to_isomer_indexes[nonzero_scan_index]
                    scores_and_key_sets_this_scan = []
                    all_unique_keys_this_scan = []
                    for ref_isomer_index in ref_isomer_indexes:
                        product_spectrum_i = isomer_index_dict_to_prod_spectra[ref_isomer_index]
                        this_scan_element = product_spectrum_i[nonzero_scan_index]
                        this_scan_score = this_scan_element[1]
                        product_key_set_i = isomer_index_dict_to_keys[ref_isomer_index]
                        scores_and_key_sets_this_scan += [[this_scan_score, product_key_set_i]]
                        for product_key_j in product_key_set_i:
                            unique_key_j = True
                            for unique_key_this_scan in all_unique_keys_this_scan:
                                if unique_key_this_scan == product_key_j:
                                    unique_key_j = False
                            if unique_key_j:
                                all_unique_keys_this_scan += [product_key_j]

                    # determining the total counts, solvable or otherwise
                    total_counts_this_scan = 0.0
                    for unique_key_this_scan in all_unique_keys_this_scan:
                        unique_key_spectrum = keys_and_finished_spectra[unique_key_this_scan]
                        key_spectrum_element = unique_key_spectrum[nonzero_scan_index]
                        key_spectrum_value = key_spectrum_element[1]
                        total_counts_this_scan += key_spectrum_value

                    # determining if the scan is solvable
                    top_solvable_key_sets = get_top_solvable_key_sets(scores_and_key_sets_this_scan)
                    if len(top_solvable_key_sets) != len(scores_and_key_sets_this_scan):
                        has_unsolvable_isomers = True
                    if len(top_solvable_key_sets) > 0:
                        solvable_scan_indexes += [nonzero_scan_index]
                        scan_index_dict_to_solvable_sets[nonzero_scan_index] = top_solvable_key_sets
                        total_solvable_counts += total_counts_this_scan
                        scan_solvable_isomer_indexes = []
                        for top_solvable_key_set in top_solvable_key_sets:
                            for isomer_index in final_isomer_indexes:
                                ref_key_set_i = isomer_index_dict_to_keys[isomer_index]
                                if ref_key_set_i == top_solvable_key_set:
                                    scan_solvable_isomer_indexes += [isomer_index]
                        scan_index_dict_to_solvable_isomer_indexes[nonzero_scan_index] = scan_solvable_isomer_indexes
                    else:
                        unsolvable_scan_indexes += [nonzero_scan_index]
                        total_unsolvable_counts += total_counts_this_scan

            if total_unsolvable_counts > 0.0:
                has_unsolvable_isomers = True

            total_counts = total_solvable_counts + total_unsolvable_counts
            proceed_threshold = total_counts * 0.95

            if total_solvable_counts < proceed_threshold:
                proceed = False

            # generating corrected counts spectra
            isomer_index_dict_to_corrected_counts = {}
            nested_dict_scan_index_to_key_to_counts = {}
            all_solvable_isomer_indexes = []
            if proceed:

                # creating scan index to oznox key to counts nested dictionary
                for solvable_scan_index in solvable_scan_indexes:
                    new_dictionary = {}
                    for unique_key in unique_keys:
                        unique_key_spectrum = keys_and_finished_spectra[unique_key]
                        spectrum_scan_value = unique_key_spectrum[solvable_scan_index][1]
                        new_dictionary[unique_key] = spectrum_scan_value
                    nested_dict_scan_index_to_key_to_counts[solvable_scan_index] = new_dictionary

                # creating corrected counts tallying dictionary
                for solvable_scan_index in solvable_scan_indexes:
                    ref_solvable_sets = scan_index_dict_to_solvable_sets[solvable_scan_index]
                    solvable_isomer_indexes = []
                    for isomer_index in final_isomer_indexes:
                        ref_key_set = isomer_index_dict_to_keys[isomer_index]
                        for ref_solvable_set in ref_solvable_sets:
                            if ref_solvable_set == ref_key_set:
                                solvable_isomer_indexes += [isomer_index]
                    for solvable_isomer_index in solvable_isomer_indexes:
                        isomer_index_dict_to_corrected_counts[solvable_isomer_index] = 0.0
                        new_isomer_index = True
                        for all_solvable_isomer_index in all_solvable_isomer_indexes:
                            if all_solvable_isomer_index == solvable_isomer_index:
                                new_isomer_index = False
                        if new_isomer_index:
                            all_solvable_isomer_indexes += [solvable_isomer_index]

                # iterating through solvable scans
                for solvable_scan_index in solvable_scan_indexes:

                    key_counts_dict = nested_dict_scan_index_to_key_to_counts[solvable_scan_index]
                    working_key_counts = copy.deepcopy(key_counts_dict)
                    solvable_isomer_indexes = scan_index_dict_to_solvable_isomer_indexes[solvable_scan_index]
                    working_key_sets = []
                    for solvable_isomer_index in solvable_isomer_indexes:
                        new_set = isomer_index_dict_to_keys[solvable_isomer_index]
                        working_key_sets += [new_set]

                    # iteratively assigning counts one unique key set at a time, reducing remaining counts accordingly
                    while len(working_key_sets) > 0:
                        result_array = find_unique_key_set_in_set_of_key_sets(working_key_sets)
                        current_key_set = result_array[0]
                        remaining_sets = result_array[1]
                        working_key_sets = copy.deepcopy(remaining_sets)
                        current_isomer_index = None
                        for solvable_isomer_index in solvable_isomer_indexes:
                            reference_set = isomer_index_dict_to_keys[solvable_isomer_index]
                            if reference_set == current_key_set:
                                current_isomer_index = solvable_isomer_index

                        # determining lowest effective counts (counts / number of double-bonds represented by this key)
                        # validating that there are still counts available for this species
                        lowest_effective_counts = -1
                        key_multiplicities = {}
                        no_zeros = True
                        unique_keys_this_set = []
                        for oznox_key_i in current_key_set:

                            new_key_this_set = True
                            for unique_key_this_set in unique_keys_this_set:
                                if unique_key_this_set == oznox_key_i:
                                    new_key_this_set = False
                            if new_key_this_set:
                                unique_keys_this_set += [oznox_key_i]

                            key_counts_i = working_key_counts[oznox_key_i]
                            key_multiplicity = 0.0
                            for oznox_key_j in current_key_set:
                                if oznox_key_i == oznox_key_j:
                                    key_multiplicity += 1.0
                            key_multiplicities[oznox_key_i] = key_multiplicity
                            effective_counts = key_counts_i / key_multiplicity
                            if effective_counts == 0:
                                no_zeros = False
                            if effective_counts < lowest_effective_counts or lowest_effective_counts == -1:
                                lowest_effective_counts = effective_counts

                        if no_zeros:  # continue if all necessary products still have counts available

                            # subtracting consumed counts.  This would be more accurate if an equation based on n-# were
                            # applied.  It is known that intensity gradually decreases as n-# increases.
                            for unique_key_this_set in unique_keys_this_set:
                                old_key_counts_i = working_key_counts[unique_key_this_set]
                                key_multiplicity = key_multiplicities[unique_key_this_set]
                                new_key_counts_i = old_key_counts_i - (lowest_effective_counts * key_multiplicity)
                                if new_key_counts_i < 0:
                                    new_key_counts_i = 0.0
                                working_key_counts[unique_key_this_set] = new_key_counts_i

                            old_isomer_counts = isomer_index_dict_to_corrected_counts[current_isomer_index]
                            new_isomer_counts = old_isomer_counts + (lowest_effective_counts * len(current_key_set))
                            isomer_index_dict_to_corrected_counts[current_isomer_index] = new_isomer_counts

            # generating corrected quantification factor
            indexes_corrected_counts_factors = []
            has_quantification = False
            if proceed:
                sum_corrected_counts = 0.0
                for solvable_isomer_index in all_solvable_isomer_indexes:
                    corrected_counts = isomer_index_dict_to_corrected_counts[solvable_isomer_index]
                    if corrected_counts >= 0.0:
                        sum_corrected_counts += corrected_counts
                for solvable_isomer_index in all_solvable_isomer_indexes:
                    corrected_counts = isomer_index_dict_to_corrected_counts[solvable_isomer_index]
                    corrected_factor = corrected_counts / sum_corrected_counts
                    if has_unsolvable_isomers:
                        corrected_factor_rounded = round(corrected_factor, 3)
                        corrected_factor_rounded = round((corrected_factor_rounded * 100.0), 2)
                        corrected_factor_string = str(corrected_factor_rounded)
                        corrected_factor_string += '%'
                    else:
                        highest_string_index = -1
                        for solvable_isomer_index_2 in all_solvable_isomer_indexes:
                            corrected_counts_2 = isomer_index_dict_to_corrected_counts[solvable_isomer_index_2]
                            corrected_factor_2 = corrected_counts_2 / sum_corrected_counts
                            if corrected_factor_2 > 0:
                                corrected_factor_2_string = str(corrected_factor_2)
                                found_nonzero = False
                                string_index = 0
                                while not found_nonzero:
                                    string_char = corrected_factor_2_string[string_index]
                                    string_index += 1
                                    if string_char != '0' and string_char != '.':
                                        found_nonzero = True
                                if highest_string_index == -1 or string_index > highest_string_index:
                                    highest_string_index = string_index
                        num_decimals = highest_string_index - 2
                        if num_decimals < 2:
                            num_decimals = 2
                        corrected_factor_rounded = round(corrected_factor, num_decimals)
                        corrected_factor_rounded = round((corrected_factor_rounded * 100), (num_decimals - 2))
                        corrected_factor_string = str(corrected_factor_rounded)
                        corrected_factor_string += '%'
                    indexes_corrected_counts_factors += [[solvable_isomer_index, corrected_counts,
                                                          corrected_factor_string]]
                if sum_corrected_counts > 0.0:
                    has_quantification = True

            # inserting calculated values into output DataFrame
            if has_quantification:
                for index_corrected_count_factor in indexes_corrected_counts_factors:
                    isomer_index = index_corrected_count_factor[0]
                    corrected_count = index_corrected_count_factor[1]
                    quant_factor = index_corrected_count_factor[2]
                    output_frame.loc[isomer_index, group_sum_corrected_counts_column] = corrected_count
                    output_frame.loc[isomer_index, group_corrected_quant_column] = quant_factor

                for isomer_index in isomer_indexes:
                    output_frame_row = output_frame.loc[isomer_index]
                    quant_factor = str(output_frame_row[group_corrected_quant_column])
                    corrected_count = str(output_frame_row[group_sum_corrected_counts_column])
                    if corrected_count == '0':
                        output_frame.loc[isomer_index, group_sum_corrected_counts_column] = ''
                    if len(corrected_count) == 0 or corrected_count.upper() == 'NAN':
                        output_frame.loc[isomer_index, group_sum_corrected_counts_column] = ''
                        output_frame.loc[isomer_index, group_corrected_quant_column] = '< 1%'
                    elif len(quant_factor) == 0 or quant_factor.upper() == 'NAN':
                        output_frame.loc[isomer_index, group_sum_corrected_counts_column] = ''
                        output_frame.loc[isomer_index, group_corrected_quant_column] = '< 1%'
                    else:
                        if quant_factor == '0.0%':
                            output_frame.loc[isomer_index, group_corrected_quant_column] = '< 1%'
                        if quant_factor == '100.0%':
                            if len(isomer_indexes) > 1:
                                output_frame.loc[isomer_index, group_corrected_quant_column] = '> 99%'
                            else:
                                output_frame.loc[isomer_index, group_corrected_quant_column] = '100%'

    temp_frame = output_frame.sort_values(by=['Parent ID', 'All Samples Max Log Score'],
                                          ascending=[True, False])
    temp_frame = temp_frame.reset_index(drop=True)
    output_frame = temp_frame.copy()
    output_name = 'OzNOx Script 7 output ' + time_stamp() + '.csv'
    output_frame.to_csv(output_name, index=False)
    print('')
    print('Output available as ' + output_name)
    print('')
    print('Script 7 finished.')
    return None


# visualize RT:counts spectrum, copy-pasted by user
def oznox_script_8():

    print('')
    print('Beginning Script 8 spectrum visualization ...')

    spectrum = input('Enter spectrum to visualize (copy-paste from .csv file): ')
    proceed = True
    spectrum_items = None
    rt_values = []
    counts_values = []
    try:
        spectrum_items = re.split(' ', spectrum)
    except Exception as e:
        print(e)
        print('Error: Spectrum entered is not formatted correctly')
        print('Compatible formatting - RT:Counts RT:Counts RT:Counts RT:Counts RT:Counts ...')
        proceed = False
    if proceed:
        for spectrum_item in spectrum_items:
            if proceed:
                try:
                    spectrum_item_split = re.split(':', spectrum_item)
                    rt_value = float(spectrum_item_split[0])
                    rt_values += [rt_value]
                    counts_value = float(spectrum_item_split[1])
                    counts_values += [counts_value]
                except Exception as e:
                    print(e)
                    print('Error: could not parse spectrum')
                    proceed = False
    if proceed:
        data_frame = pd.DataFrame({'x': rt_values, 'height': counts_values})
        data_frame = data_frame.sort_values(by=['height'], ascending=[False])
        data_frame = data_frame.reset_index()
        data_frame.drop(['index'], axis=1)
        rt_values.sort()
        smallest_delta_rt = -1
        previous_rt_value = -1
        for rt_value in rt_values:
            if smallest_delta_rt == -1 and previous_rt_value == -1:
                previous_rt_value = rt_value
            else:
                rt_delta = abs(rt_value - previous_rt_value)
                previous_rt_value = rt_value
                if smallest_delta_rt == -1 or rt_delta < smallest_delta_rt:
                    smallest_delta_rt = rt_delta
        plt.bar(x=data_frame.x, height=data_frame.height, width=smallest_delta_rt)
        plt.xlabel('RT')
        plt.ylabel('Counts')
        plt.title('RT:Counts Spectrum')
        print('')
        print('Spectrum visualized.  Close the spectrum to return to main menu.')
        plt.show()
        return None


# main() contains the user inputs and OzNOx data processing workflow with references to the other functions above
def main():

    # Version number and description
    version_number_string = 'v1.0'
    version_description_string = 'Publication Version'

    # Section below is printed on launching OzNOx Companion.  Printed information includes version and documentation.
    print('')
    print('***********************************************************************************************************')
    print('    ************                      ***         *     ************                       ************    ')
    print('    *          *                      *  *        *     *          *                       *               ')
    print('    *          *     ************     *   *       *     *          *     *           *     *               ')
    print('    *          *               *      *    *      *     *          *       *       *       *               ')
    print('    *          *             *        *     *     *     *          *         *   *         *               ')
    print('    *          *           *          *      *    *     *          *           *           *               ')
    print('    *          *         *            *       *   *     *          *         *   *         *               ')
    print('    *          *       *              *        *  *     *          *       *       *       *               ')
    print('    ************     ************     *         ***     ************     *           *     ************ *  ')
    print('***********************************************************************************************************')
    print('')
    print('OzNOx Companion ' + version_number_string + ' : ' + version_description_string)
    print('Annotation of Unsaturated Lipid Double-Bonds by LC-OzNOxESI-MS/MS')
    print('')
    print('Developed at the Center for Translational Biomedical Research, University of North Carolina Greensboro')
    print('600 Laureate Way')
    print('Suite 2203')
    print('Kannapolis, NC 28081')
    print('USA')
    print('')
    print('Developed by: Smith, R.; Omar, A.; Zhang, Q.')
    print('Development supported by Diabetes and Digestive and Kidney Diseases of the National Institutes of Health '
          'Award No. R01DK123499')
    print('')
    print('Developed with python packages:')
    print('copy')
    print('datetime')
    print('math log10')
    print('matplotlib pyplot')
    print('numpy')
    print('os')
    print('pandas')
    print('re')
    print('')
    print('Please cite as: (Citation will be updated with publication.)')
    print('Smith, R.; Omar, A.; Mulani, F.; Zhang, Q. Anal. Chem. 2025, 97 (3), 1879-1888.')
    print('')
    print('For usage and trouble-shooting, refer to ReadMe file available at: github.com/QibinZhangLab/OzNOx-Companion')
    print('Email questions to: q_zhang2@uncg.edu and CC: rasmith12@uncg.edu and amalsagheer@uncg.edu')
    print('')
    print('*** Known issue: Opening .csv in Excel and converting the data causes workflow instability')
    print('*** Solution: Open the file and delete the first blank row.  Then, save and run again.')
    print('')
    print('*** Known issue: TG regioisomer relative quantification is unreliable for species with heterogeneous acyls')
    print('*** Solution: Utilize only Scheme 1 OzNOx fragmentation. A future update will provide a better fix.')
    print('')
    print('*** Known issue: sphingoid base double-bonds (Ex n-14 in d18:1) have low reactivity, resulting in'
          ' inaccurate regioisomer quantification.')
    print('*** Solution: Do not use script 7 output for investigation of sphingoid base-containing classes. Perform '
          ' manual correction, quantification from script 5 or 6 outputs.')

    # Persistent variables
    script_1_finished = False
    project_directory = None
    class_definitions_frame = None
    lcms_annotations_frame = None

    # Begin script-running loop
    while True:
        print('')
        print('Which script would you like to run?')
        print('Script 1: Select project folder and load/update LC-MS annotations, parameter files')
        print('Script 2: Process LC-MS annotations for manual RT-based validation/rejection')
        print('Script 3: Output a PRM list for LC-OzNOxESI-MS2 for your LC-MS annotations')
        print('Script 4: Convert .txt LC-MS and PRM LC-MS2 data to compatible .csv')
        print('Script 5: Search for presumptive OzID MS1 and OzNOx MS2 double-bond product species')
        print('Script 6: Unite and organize Script 5 output(s)')
        print('Script 7: (Prototype) double-bond annotation and regioisomer quantification with Script 6 output')
        print('Script 8: Visualize a text-based spectrum from Script 5, 6, or 7 output')
        input_done = False
        while not input_done:
            user_input = input('Run Script Number: ')
            if user_input == '1':
                input_done = True
                project_directory, class_definitions_frame, lcms_annotations_frame = oznox_script_1()
                os.chdir(project_directory)
                script_1_finished = True
            elif user_input == '2':
                if script_1_finished:
                    input_done = True
                    oznox_script_2(project_directory, lcms_annotations_frame)
                else:
                    print('Error: You must run Script 1 before Script 2.')
            elif user_input == '3':
                if script_1_finished:
                    input_done = True
                    oznox_script_3(project_directory, class_definitions_frame, lcms_annotations_frame)
                else:
                    print('Error: You must run Script 1 before Script 3.')
            elif user_input == '4':
                input_done = True
                oznox_script_4()
            elif user_input == '5':
                if script_1_finished:
                    input_done = True
                    oznox_script_5(project_directory, class_definitions_frame, lcms_annotations_frame)
                    os.chdir(project_directory)
                else:
                    print('Error: You must run Script 1 before Script 5.')
            elif user_input == '6':
                if script_1_finished:
                    input_done = True
                    oznox_script_6(lcms_annotations_frame)
                else:
                    print('Error: You must run Script 1 before Script 6.')
            elif user_input == '7':
                input_done = True
                oznox_script_7()
            elif user_input == '8':
                input_done = True
                oznox_script_8()
            else:
                print('Error: Invalid entry.  Enter just the number of the Script you wish to run.')


main()
