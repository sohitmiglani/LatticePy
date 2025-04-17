import sys
import copy
import math
import plotly
import random
import numpy as np
import pandas as pd
import seaborn as sns 
import networkx as nx
import plotly.io as pio
from .objects import amino_acid, linker_bead
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sns.set()
system_random = random.SystemRandom()
pio.renderers.default = 'iframe_connected'

class lattice():
    def __init__(self, bound, energy=2.5, beta=0, lattice_type='simple_cubic', record_moves=False, allow_cluster_move=True):
        self.lattice_type = lattice_type
        self.record_moves = record_moves
        self.bound = bound
        self.space = dict()
        self.linker_space = dict()
        self.start = dict()
        self.linker_start = dict()
        self.last = dict()
        self.energy = 0
        self.native_contacts = 0
        self.beta = beta
        self.n_mcmc = 0
        self.length_of_polymer = 0
        self.energy_records = []
        self.beta_records = []
        self.native_contacts_records = []
        self.move_records = []
        self.acceptance_records = []
        self.records = []
        self.n_polymers = 0
        self.individual_energy = []
        self.individual_energy_records = []
        self.individual_ncs_records = []
        self.allow_cluster_move = allow_cluster_move
        self.energy_step = -energy
        self.previous_space = []
        self.previous_objects = []
        self.previous_start = []
        self.previous_last = []
            
    def periodic_coordinate(self, coordinates):
        return [round(coordinates[0]%(self.bound +1)), 
                round(coordinates[1]%(self.bound +1)), 
                round(coordinates[2]%(self.bound +1))]
                
    def find_valid_neighbors(self, aa):
        if self.lattice_type == 'simple_cubic':
            neighbors = []
            for i in range(3):
                neighbor = aa.coordinates.copy()
                neighbor[i] += 1
                neighbor_rev = aa.coordinates.copy()
                neighbor_rev[i] += -1
                neighbors.append(neighbor)
                neighbors.append(neighbor_rev)
            neighbors = [self.periodic_coordinate(neighbor) for neighbor in neighbors]
            neighbors = [neighbor for neighbor in neighbors if str(neighbor) not in [str(aa.next),str(aa.previous)]]
            return neighbors

    def find_valid_neighbors_linker(self, linker):
        if self.lattice_type == 'simple_cubic':
            neighbors = []
            for i in range(3):
                neighbor = linker.coordinates.copy()
                neighbor[i] += 1
                neighbor_rev = linker.coordinates.copy()
                neighbor_rev[i] += -1
                neighbors.append(neighbor)
                neighbors.append(neighbor_rev)
            neighbors = [self.periodic_coordinate(neighbor) for neighbor in neighbors]
            neighbors = [neighbor for neighbor in neighbors if str(neighbor) != str(linker.neighbor)]
            return neighbors

    def validate_chain(self):
        for i in range(self.n_polymers):
            if i not in list(self.start.values()):
                raise RuntimeError('Polymer numbers not distinct in start')
            if i not in list(self.last.values()):
                raise RuntimeError('Polymer numbers not distinct in last')
        
        for start in list(self.start.keys()):
            i = 0
            records = []
            start = self.space[start]
            while i < self.length_of_polymer:
                if max(start.coordinates) > self.bound or min(start.coordinates) < 0:
                    raise RuntimeError('Periodic Boundary Conditions have been violated.')
                records.append(str(start.coordinates))
                i+= 1
                next_coordinate = copy.copy(start.next)
                if next_coordinate is None:
                    if i==self.length_of_polymer:
                        break
                    else:
                        raise RuntimeError('The chain has broken. The polymer does not have the specified number of residues.')
                if str(next_coordinate) in records:
                    raise RuntimeError('The chain has two residues on the same site')

                if math.dist(start.coordinates, start.next) not in [1, self.bound]:
                    raise RuntimeError('The distance between neighbors is not valid. The chain has broken.')

                start = self.space[str(next_coordinate)]
                
        if len(self.space) != self.n_polymers*self.length_of_polymer:
            raise RuntimeError('There is an overlap between amino acids of different polymers.')
        return True

    def move_success(self, deltaE):
        if system_random.random() < np.exp(-deltaE*self.beta):
            return True
        else:
            return False
    
    def update_system_energy(self):
        energy = 0
        individual_ncs = []
        individual_energies = []
        
        for last in list(self.last.keys()):
            indiv_ncs = 0
            indiv_energy = 0
            aa = self.space[str(last)]
            while True:
                coords = aa.coordinates
                if str(coords) in self.linker_space:
                    indiv_ncs += 1
                    indiv_energy += self.energy_step
                    energy += self.energy_step
                    
                if aa.previous is not None:
                    aa = self.space[str(aa.previous)]
                else:
                    break
            individual_ncs.append(indiv_ncs)
            individual_energies.append(round(indiv_energy,2))
        
        self.individual_ncs = individual_ncs
        self.individual_energy = individual_energies
        self.native_contacts = sum(individual_ncs)
        self.energy = round(energy,2)

    def measure_system_energy(self, new_space, new_starts, new_lasts, nearest_neighbors=True):
        energy = 0
        individual_ncs = []
        individual_energies = []
        
        for last in list(new_lasts.keys()):
            indiv_ncs = 0
            indiv_energy = 0
            aa = new_space[str(last)]
            while True:
                coords = aa.coordinates
                if str(coords) in self.linker_space:
                    indiv_ncs += 1
                    indiv_energy += self.energy_step
                    energy += self.energy_step
                
                if aa.previous is not None:
                    aa = new_space[str(aa.previous)]
                else:
                    break
            individual_ncs.append(indiv_ncs)
            individual_energies.append(round(indiv_energy,2))

        return individual_ncs, individual_energies, sum(individual_ncs), round(energy,2)

    def find_subsystem_energy(self, originals, replacements, type):
        original_energy = 0
        original_indiv_ncs = 0
        new_energy = 0
        new_indiv_ncs = 0

        if type == 'polymer':
            for coords in originals:
                if str(coords) in self.linker_space:
                    original_energy += self.energy_step
                    original_indiv_ncs += 1
        
            for coords in replacements:
                if str(coords) in self.linker_space:
                    new_energy += self.energy_step
                    new_indiv_ncs += 1
                    
        elif type == 'linker':
            for coords in originals:
                if str(coords) in self.space:
                    original_energy += self.energy_step
                    original_indiv_ncs += 1
        
            for coords in replacements:
                if str(coords) in self.space:
                    new_energy += self.energy_step
                    new_indiv_ncs += 1
        
        return round(new_energy - original_energy,2), new_indiv_ncs - original_indiv_ncs
    
    def move_chain(self, originals, replacements, inflection_point, start_or_end, type='polymer'):
        if type == 'polymer':
            for i in range(len(originals)):
                if str(replacements[i]) in self.linker_space:
                    if self.space[str(originals[i])].polarity != self.linker_space[str(replacements[i])].polarity:
                        if self.record_moves:
                            self.acceptance_records.append(0)
                        return False
         
        deltaE, deltaNC = self.find_subsystem_energy(originals, replacements, type)
        self.validate_chain()
        
        if self.move_success(deltaE):
            self.energy += round(deltaE,2)
            self.native_contacts += deltaNC
            if self.record_moves:
                self.acceptance_records.append(1)
            if type == 'polymer':
                polymer_id = self.space[originals[0]].polymer
                self.individual_ncs[polymer_id] += deltaNC
                self.individual_energy[polymer_id] += round(deltaE,2)
                #print(originals)
                #print(replacements)
                #print(start_or_end)
                #self.previous_space.append(self.space.copy())
                #self.previous_start.append(self.start.copy())
                #self.previous_last.append(self.last.copy())
                
                all_objects = []
                
                for aa_step in range(len(originals)):
                    original = copy.copy(originals[aa_step])
                    replacement = copy.copy(replacements[aa_step])
                    replacement_int = [int(i) for i in replacement.strip('][').split(', ')].copy()
                    original_aa = copy.copy(self.space[original])
                    original_aa.coordinates = replacement_int.copy()
                    if start_or_end == 0:
                        if aa_step != 0: 
                            original_aa.next = [int(i) for i in replacements[aa_step-1].strip('][').split(', ')].copy()
                        if original_aa.previous is not None and len(replacements) > 1 and aa_step < len(originals)-1:
                            original_aa.previous = [int(i) for i in replacements[aa_step+1].strip('][').split(', ')].copy()
                    elif start_or_end == 1:
                        if aa_step != 0:
                            original_aa.previous = [int(i) for i in replacements[aa_step-1].strip('][').split(', ')].copy()
                        if original_aa.next is not None and len(replacements) > 1 and aa_step < len(originals)-1:
                            original_aa.next = [int(i) for i in replacements[aa_step+1].strip('][').split(', ')].copy()
                    all_objects.append(copy.copy(original_aa))

                #self.previous_objects = all_objects
                
                for aa_step in range(len(originals)):
                    original = originals[aa_step]
                    original_int = [int(i) for i in original.strip('][').split(', ')].copy()
                    replacement = replacements[aa_step]
                    replacement_int = [int(i) for i in replacement.strip('][').split(', ')].copy()
                    replacement_object = copy.copy(all_objects[aa_step])
                    if aa_step ==0:
                        if original in self.last.keys():
                            del self.last[copy.copy(original)]
                            self.last[replacement] = polymer_id
                        elif original in self.start.keys():
                            del self.start[copy.copy(original)]
                            self.start[replacement] = polymer_id
                        
                        if len(originals) != self.length_of_polymer:
                            if start_or_end == 0:
                                self.space[str(inflection_point)].previous = replacement_int.copy()
                            else:
                                self.space[str(inflection_point)].next = replacement_int.copy()
                    if aa_step == len(originals)-1:
                        if original in self.last.keys():
                            del self.last[copy.copy(original)]
                            self.last[replacement] = polymer_id
                        elif original in self.start.keys():
                            del self.start[copy.copy(original)]
                            self.start[replacement] = polymer_id
                        if start_or_end == 0:
                            if replacement_object.previous is not None:
                                self.space[str(replacement_object.previous)].next = replacement_int
                        else:
                            if replacement_object.next is not None:
                                self.space[str(replacement_object.next)].previous = replacement_int
                    self.space[replacement] = replacement_object
                    if original not in replacements:
                        del self.space[original]
                return True
            elif type == 'linker':
                polymer_id = self.linker_space[str(originals[0])].polymer

                if start_or_end == 0:
                    del self.linker_start[str(originals[0])]
                    self.linker_start[str(replacements[0])] = polymer_id

                if len(originals) == 1:
                    original_object = self.linker_space[str(originals[0])]
                    original_object.coordinates = replacements[0]
                    self.linker_space[str(replacements[0])] = original_object
                    del self.linker_space[str(originals[0])]
                    self.linker_space[str(original_object.neighbor)].neighbor = replacements[0]

                else:
                    original_object = self.linker_space[str(originals[0])]
                    original_object.coordinates = replacements[0]
                    original_object.neighbor = replacements[1]
                    
                    original_object2 = self.linker_space[str(originals[1])]
                    original_object2.coordinates = replacements[1]
                    original_object2.neighbor = replacements[0]
                    
                    del self.linker_space[str(originals[0])]
                    del self.linker_space[str(originals[1])]
                    
                    self.linker_space[str(replacements[0])] = original_object
                    self.linker_space[str(replacements[1])] = original_object2

                    if start_or_end == 1:
                        del self.linker_start[str(originals[1])]
                        self.linker_start[str(replacements[1])] = polymer_id

                for original in originals:
                    if str(original) in self.space:
                        polymer_id = self.space[str(original)].polymer
                        self.individual_ncs[polymer_id] -= 1
                        self.individual_energy[polymer_id] -= 1

                for replacement in replacements:
                    if str(replacement) in self.space:
                        polymer_id = self.space[str(replacement)].polymer
                        self.individual_ncs[polymer_id] += 1
                        self.individual_energy[polymer_id] += 1
                return True
        else:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False
            
    def add_polymer(self, polymer, n_polymers=1, placement='straight'):
        length = len(polymer)
        polymers_placed = 0

        while polymers_placed < n_polymers:
            polymer_id = len(self.start)
            new_coords = None
            if len(self.start) > 0:
                new_coords = system_random.choice(list(self.start.keys()))
                new_coords = [int(i) for i in new_coords.strip('][').split(', ')].copy()
                axis_start = system_random.choice([0,1,2])
                change = system_random.choice([1,2])
                new_coords[axis_start] += system_random.choice([+change, -change])
            else:
                x = system_random.randint(0, self.bound)
                y = system_random.randint(0, self.bound)
                z = system_random.randint(0, self.bound)
                new_coords = [x, y, z]
            
            new_coords = self.periodic_coordinate(new_coords)
            all_placements = [] 
            if str(new_coords) not in self.space:
                if placement == 'straight':
                    axis = system_random.choice([0,1,2])
                    direction = system_random.choice([1, -1])
                    for i in range(1, length):
                        tries = 0 
                        while tries < 5:
                            tries += 1
                            new = new_coords.copy()
                            new[axis] += direction*i
                            new = self.periodic_coordinate(new)
                            if str(new) not in self.space:
                                all_placements.append(new.copy())
                                break
                elif placement == 'randomly':
                    coordinates = new_coords.copy()
                    for i in range(length-1):
                        tries = 0 
                        while tries < 5:
                            tries += 1
                            next_coordinates = coordinates.copy()
                            axis = system_random.randint(0,2)
                            next_coordinates[axis] += system_random.randint(-1,1)
                            next_coordinates = self.periodic_coordinate(next_coordinates)
                            if str(next_coordinates) not in self.space and next_coordinates not in all_placements:
                                all_placements.append(next_coordinates.copy())
                                break
                        coordinates = next_coordinates.copy()
            if len(all_placements) == length-1:
                self.start[str(new_coords)] = polymer_id
                self.space[str(new_coords)] = amino_acid(polymer[0],new_coords, polymer_id)
                current_aa = self.space[str(new_coords)]
                for i in range(length-1):
                    next_coordinates = all_placements[i]
                    self.space[str(current_aa.coordinates)].next = next_coordinates
                    self.space[str(next_coordinates)] = amino_acid(polymer[i+1], next_coordinates, polymer_id)
                    self.space[str(next_coordinates)].previous = current_aa.coordinates.copy()
                    current_aa = self.space[str(next_coordinates)]
                polymers_placed += 1
                self.length_of_polymer = length
                self.n_polymers += 1
                self.last[str(next_coordinates)] = polymer_id

    def add_linker(self, n_polymers=1):

        polymers_placed = 0
        while polymers_placed < n_polymers:
            new_coords = None
            valid_neighbor = None
            valid_path=False
            
            while not valid_path:
                x = system_random.randint(0, self.bound)
                y = system_random.randint(0, self.bound)
                z = system_random.randint(0, self.bound)
                new_coords = [x, y, z]
                
                if str(new_coords) not in self.linker_space:
                    if str(new_coords) not in self.space or self.space[str(new_coords)].polarity== 1:
                        valid_path=True
                    else:
                        continue

                if valid_path:
                    neighbors = []
                    for i in range(3):
                        neighbor = new_coords.copy()
                        neighbor[i] += 1
                        neighbor_rev = new_coords.copy()
                        neighbor_rev[i] += -1
                        neighbors.append(neighbor)
                        neighbors.append(neighbor_rev)
                    neighbors = [self.periodic_coordinate(neighbor) for neighbor in neighbors]
                    random.shuffle(neighbors)
                    
                    valid_path = False
                    for neighbor in neighbors:
                        if str(neighbor) not in self.linker_space:
                            if str(neighbor) not in self.space or self.space[str(neighbor)].polarity== -1:
                                valid_neighbor = neighbor.copy()
                                valid_path=True
                                break

            self.linker_start[str(new_coords)] = polymers_placed
            self.linker_space[str(new_coords)] = linker_bead(polarity=1, coordinates=new_coords, neighbor=valid_neighbor, polymer_id=polymers_placed)
            self.linker_space[str(valid_neighbor)] = linker_bead(polarity=-1, coordinates=valid_neighbor, neighbor=new_coords, polymer_id=polymers_placed)
            polymers_placed += 1
                                

    def end_move(self):
        if self.record_moves:
            self.move_records.append('end_move')
        start_or_end=system_random.choice([0,1])
        coordinates = system_random.choice([ list(self.start.keys()), list(self.last.keys())][start_or_end])
        coordinates = self.periodic_coordinate([int(i) for i in coordinates.strip('][').split(', ')])
        current_aa = copy.copy(self.space[str(coordinates)])
        next_coordinates = None
        tries = 0

        while tries < 5:
            tries += 1
            next_coordinates = coordinates.copy()  
            back = None
            if start_or_end == 0:
                back = current_aa.next.copy()
            else:
                back = current_aa.previous.copy()
            axis = None
            for i in range(3):
                if back[i] !=  next_coordinates[i]:
                    next_coordinates[i] = copy.copy(back[i])
                    axis = i
            next_axis = system_random.choice([i for i in [0,1,2] if i != axis])
            direction = system_random.choice([1, -1])
            next_coordinates[next_axis] += direction
            next_coordinates = self.periodic_coordinate(next_coordinates)
            if str(next_coordinates) not in self.space:
                return self.move_chain([str(coordinates)], [str(next_coordinates)], back, start_or_end, type='polymer')
        
        if self.record_moves:
            self.acceptance_records.append(0)

    def crankshaft_move(self):
        if self.record_moves:
            self.move_records.append('crankshaft')
        aa = self.space[system_random.choice(list(self.start.keys()))]
        tries = 0
        while tries < self.length_of_polymer-3:
            tries += 1
            axis_1 = None
            axis_2 = None
            axis_3 = None
            coordinates = aa.coordinates.copy()
            first_coordinates = aa.next.copy()
            second_coordinates = self.space[str(first_coordinates)].next.copy()
            third_coordinates = self.space[str(second_coordinates)].next.copy()
            first_direction = None
            back_direction = None

            for i in range(3):
                if coordinates[i] != first_coordinates[i]:
                    first_direction = first_coordinates[i] - coordinates[i]
                    axis_1 = i
            for j in range(3):
                if second_coordinates[j] != third_coordinates[j]:
                    back_direction = third_coordinates[j] - second_coordinates[j]
                    axis_3 = j
            for k in range(3):
                if first_coordinates[k] != second_coordinates[k]:
                    axis_2 = k
            
            if axis_1 == axis_3 and axis_1 != axis_2 and first_direction*back_direction == -1:     
                axis_of_replacement = [i for i in [0,1,2] if i not in [axis_1, axis_2]][0]              
                for direction in [1, -1]:
                    first_try = first_coordinates.copy()
                    second_try = second_coordinates.copy()
                    first_try[axis_1] += back_direction
                    second_try[axis_1] += back_direction
                    first_try[axis_of_replacement] += direction
                    second_try[axis_of_replacement] += direction
                    first_try = self.periodic_coordinate(first_try)
                    second_try = self.periodic_coordinate(second_try)
                    
                    if str(first_try) not in self.space and str(second_try) not in self.space:
                        return self.move_chain([str(second_coordinates), str(first_coordinates)], [str(second_try), str(first_try)], third_coordinates, 0, type='polymer')
                    else:
                        continue
            if aa.next is not None:
                aa = self.space[str(aa.next.copy())]

        if self.record_moves:
            self.acceptance_records.append(0)

    def corner_move(self):
        if self.record_moves:
            self.move_records.append('corner_move')
        polymer_id=system_random.choice(range(len(self.start.keys())))
        start_or_end=system_random.choice([0,1])
        deltaE = 0
        coordinates = [ list(self.start.keys()), list(self.last.keys())][start_or_end][polymer_id] 
        coordinates = [int(i) for i in coordinates.strip('][').split(', ')]
        current_aa = self.space[str(coordinates)]
        if start_or_end == 0:
            next_coordinates = current_aa.next.copy()
        else:
            next_coordinates = current_aa.previous.copy()
        first_rep = coordinates.copy()
        second_rep = next_coordinates.copy()
        tries = 0
        while tries < 5:
            tries += 1
            second_rep = next_coordinates.copy()
            if start_or_end == 0:
                third = self.space[str(second_rep)].next.copy()
            else:
                third = self.space[str(second_rep)].previous.copy()
            first_direction = 0
            axis = None
            for i in range(3):
                if second_rep[i] !=  third[i]:
                    first_direction = third[i] - second_rep[i]
                    second_rep[i] = third[i]
                    axis = i
            next_axis = system_random.choice([i for i in [0,1,2] if i != axis])
            next_direction = system_random.choice([1, -1])
            second_rep[next_axis] += next_direction
            first_rep = coordinates.copy()
            first_rep[axis] += first_direction
            first_rep[next_axis] += next_direction
            
            first_rep = self.periodic_coordinate(first_rep)
            second_rep = self.periodic_coordinate(second_rep)
            
            if str(first_rep) not in self.space and str(second_rep) not in self.space:
                originals = [str(next_coordinates), str(coordinates)]
                replacements = [str(second_rep), str(first_rep)]
                return self.move_chain(originals, replacements, third, start_or_end, type='polymer')
            
        if tries == 5:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

    def corner_move_anywhere(self):
        if self.record_moves:
            self.move_records.append('corner_anywhere')
        polymer_id=system_random.choice(range(len(self.start.keys())))
        start_or_end=system_random.choice([0,1])
        coordinates = [ list(self.start.keys()), list(self.last.keys())][start_or_end][polymer_id]
        coordinates = [int(i) for i in coordinates.strip('][').split(', ')]
        steps = system_random.randint(2, self.length_of_polymer-2)
        all_selections = []

        if start_or_end == 0:
            for step in range(steps):
                coordinates = self.space[str(coordinates)].next.copy()
                all_selections.append(coordinates)
        else:
            for step in range(steps):
                coordinates = self.space[str(coordinates)].previous.copy()
                all_selections.append(coordinates)

        inflection_point = system_random.choice(all_selections)
        valid_path = False
        tries = 0
        to_be_replaced = {}
        replacements = {}
        
        while not valid_path and tries < 5:
            tries += 1
            if start_or_end == 0:
                first_aa = self.space[str(inflection_point)].previous.copy()
                original = first_aa.copy()
            else:
                first_aa = self.space[str(inflection_point)].next.copy()
                original = first_aa.copy()
                
            first_axis = None
            first_direction = 0
            for i in range(3):
                if first_aa[i] != copy.copy(inflection_point[i]):
                    first_direction = inflection_point[i] - first_aa[i]
                    first_aa[i] = copy.copy(inflection_point[i])
                    first_axis = i
            
            new_tries = 0
            next_axis = None
            next_direction = None
            while new_tries < 5:
                new_tries += 1
                next_axis = system_random.choice([i for i in [0,1,2] if i != first_axis])
                next_direction = system_random.choice([1, -1])
                first_aa[next_axis] += next_direction
                if str(first_aa) not in self.space:
                    break
                    
            to_be_replaced = {}
            to_be_replaced_list = []
            replacements = {}
            
            while original is not None:
                to_be_replaced[str(original)] = 1
                to_be_replaced_list.append(original)
                if start_or_end == 0:
                    original = copy.copy(self.space[str(original)].previous)
                else:
                    original = copy.copy(self.space[str(original)].next)
            
            i = 0
            while i < len(to_be_replaced.keys()):
                original = to_be_replaced_list[i].copy()
                new_position = original.copy()
                new_position[first_axis] += first_direction
                new_position[next_axis] += next_direction
                new_position = self.periodic_coordinate(new_position)
                if str(new_position) not in self.space:
                    replacements[str(new_position.copy())] = 1
                    i += 1
                    valid_path = True
                else:
                    valid_path = False
                    break

        if valid_path:
            return self.move_chain( list(to_be_replaced.keys()), list(replacements.keys()), inflection_point, start_or_end, type='polymer')
        else:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

    def corner_flip(self):
        if self.record_moves:
            self.move_records.append('corner_flip')
        done = False
        tries = 0
        start=system_random.choice(list(self.start.keys()))
        center_coordinates = self.space[start].next.copy()
        replacement = None

        while self.space[str(center_coordinates)].next is not None and tries < 5:
            tries += 1
            before = self.space[str(center_coordinates)].previous
            after = self.space[str(center_coordinates)].next

            before_axis = None
            first_direction = 0
            for i in range(3):
                if before[i] != center_coordinates[i]:
                    first_direction = before[i] - center_coordinates[i]
                    before_axis = i

            next_axis = None
            next_direction = 0
            for j in range(3):
                if after[j] != center_coordinates[j]:
                    next_direction = after[j] - center_coordinates[j]
                    next_axis = j

            if before_axis != next_axis:
                replacement = center_coordinates.copy()
                replacement[before_axis] += first_direction
                replacement[next_axis] += next_direction
                replacement = self.periodic_coordinate(replacement)
                if str(replacement) not in self.space:
                    return self.move_chain([str(center_coordinates)], [str(replacement)], after, 0, type='polymer')
            center_coordinates = self.space[str(center_coordinates)].next
        if self.record_moves:
            self.acceptance_records.append(0)
        
    def reptation_move(self):
        if self.record_moves:
            self.move_records.append('reptation')
        polymer_id=system_random.choice(range(len(self.start.keys())))
        start_or_end=system_random.choice([0,1])
        coordinates = [ list(self.start.keys()), list(self.last.keys())][start_or_end][polymer_id]
        originals = []
        replacements = []
        originals.append(copy.copy(coordinates))
        coordinates = [int(i) for i in coordinates.strip('][').split(', ')]
        tries = 0
        while tries < 5:
            tries += 1
            main_axis = system_random.choice([0, 1, 2])
            main_direction = system_random.choice([1, -1])
            rep = coordinates.copy()
            rep[main_axis] += main_direction
            rep = self.periodic_coordinate(rep)
            if str(rep) not in self.space:
                replacements.append(str(rep))
                main = coordinates.copy()
                while True:
                    if start_or_end == 0:
                        main = self.space[str(main)].next
                    else:
                        main = self.space[str(main)].previous
                    if main is None:
                        break
                    replacements.append(originals[-1])
                    originals.append(str(main))
   
                originals.reverse()
                replacements.reverse()
                return self.move_chain(originals, replacements, None, start_or_end, type='polymer')
        self.acceptance_records.append(0)
        return False
    
    def rotation_move(self):
        if self.record_moves:
            self.move_records.append('rotation_move')
        
        def rotate(center, point, angle):
            c1, c2 = center
            p1, p2 = point
            nx = c1 + math.cos(angle) * (p1 - c1) - math.sin(angle) * (p2 - c2)
            ny = c2 + math.sin(angle) * (p1 - c1) + math.cos(angle) * (p2 - c2)
            return round(nx), round(ny)
    
        start_point = system_random.choice(list(self.start.keys()))
        all_records = []
        point = [int(i) for i in start_point.strip('][').split(', ')]
        while point is not None:
            aa = self.space[str(point)]
            all_records.append(aa.coordinates)
            point = copy.copy(aa.next)
        center = self.periodic_coordinate(np.mean(np.array(all_records), axis=0))
        all_replacements = []
        tries = 0
        while len(all_records) != len(all_replacements) and tries < 5:
            tries += 1
            all_replacements = []
            first_axis = system_random.choice([0,1,2])
            second_axis = system_random.choice([i for i in [0,1,2] if i != first_axis])
            angle = math.radians(system_random.choice([-90, 90]))
            for coordinates in all_records:
                rep = coordinates.copy()
                n1, n2 = rotate([center[first_axis], center[second_axis]], [rep[first_axis], rep[second_axis]], angle)
                rep[first_axis] = n1
                rep[second_axis] = n2
                rep = self.periodic_coordinate(rep)
                if str(rep) not in self.space or rep in all_records:
                    all_replacements.append(str(rep))
        all_records = [str(i) for i in all_records]
        if len(all_records) == len(all_replacements):
            return self.move_chain(all_records, all_replacements, None, 1, type='polymer')
        else:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

    def transform_move(self):
        if self.record_moves:
            self.move_records.append('transform_move')
        valid = False
        originals = []
        replacements = []
        tries = 0
        
        while not valid and tries < 5:
            tries += 1
            valid = False
            point = system_random.choice(list(self.start.keys()))
            polymer_id = self.space[point].polymer
            point = [int(i) for i in point.strip('][').split(', ')]
            originals = []
            replacements = []
            
            axis = system_random.choice([0, 1, 2])
            direction = system_random.choice([1, -1])
            
            while point is not None:
                originals.append(str(point))
                rep = point.copy()
                rep[axis] += direction
                rep = self.periodic_coordinate(rep)
                if str(rep) not in self.space or self.space[str(rep)].polymer == polymer_id:
                    valid = True
                    replacements.append(str(rep))
                else:
                    valid = False
                    break
                point = self.space[str(point)].next
            if valid:
                break
        
        if not valid:
            if self.record_moves:
                self.acceptance_records.append(0)
                return False
        else:
            return self.move_chain(originals, replacements, None, 1, type='polymer')

    def complexes(self):
        complexes = nx.Graph()
        for first in list(self.start.keys()):
            next = copy.copy(first)
            while next is not None:
                aa = self.space[str(next)]
                neighbors = self.find_valid_neighbors(aa)
                neighbors = [neighbor for neighbor in neighbors if str(neighbor) in self.space]
                for neighbor in neighbors:
                    if (aa.polymer, self.space[str(neighbor)].polymer) not in complexes.edges:
                        complexes.add_edge(aa.polymer, self.space[str(neighbor)].polymer)
                next = aa.next
        return complexes
        
    def cluster_move(self):
        if self.record_moves:
            self.move_records.append('cluster move')
        clusters = []
        for cluster in nx.connected_components(self.complexes()):
            clusters.append(list(cluster))

        if len(clusters) == 1:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

        cluster_choice = system_random.choice(clusters)
        direction = system_random.choice([0,1,2])
        step = system_random.choice([1, -1])
        new_space = dict()
        new_starts = copy.deepcopy(self.start)
        new_lasts = copy.deepcopy(self.last)
    
        for polymer in range(self.n_polymers):
            main = list(self.start.keys())[list(self.start.values()).index(polymer)]
            if polymer not in cluster_choice:
                for i in range(self.length_of_polymer):
                    object = copy.deepcopy(self.space[main])
                    new_space[main] = object
                    main = str(copy.copy(object.next))
            else:
                for i in range(self.length_of_polymer):
                    object = copy.deepcopy(self.space[main])
                    original_coordinates = copy.copy(object.coordinates)
                    original_next = copy.copy(object.next)
                    object.coordinates[direction] += step
                    object.coordinates = self.periodic_coordinate(object.coordinates)
                    if i == 0:
                        if polymer in list(new_starts.values()):
                            del new_starts[str(original_coordinates)]
                        new_starts[str(object.coordinates)] = polymer
                    else:
                        object.previous[direction] += step
                        object.previous = self.periodic_coordinate(object.previous)
                    
                    if i == 26:
                        if polymer in list(new_lasts.values()):
                            del new_lasts[str(original_coordinates)]
                        new_lasts[str(object.coordinates)] = polymer
                    else:
                        object.next[direction] += step
                        object.next = self.periodic_coordinate(object.next)
                    new_space[str(object.coordinates)] = object
                    main = str(original_next)

        if len(new_starts) != len(new_lasts):
            print(direction, step)
            print(cluster_choice)
            print(new_starts)
            print(new_lasts)
            print(new_space)
            sys.exit('problem problem')

        individual_ncs, individual_energies, native_contacts, energy = self.measure_system_energy(new_space, new_starts, new_lasts)

        deltaE = energy - self.energy
        if self.move_success(deltaE):
            if self.record_moves:
                self.acceptance_records.append(1)
            self.space = new_space
            self.start = new_starts
            self.last = new_lasts
            self.individual_ncs = individual_ncs
            self.individual_nchcs = individual_nchcs
            self.individual_energy = individual_energies
            self.native_contacts = native_contacts
            self.non_covalent_hydrophobic_contacts = hydrophobic_contacts
            self.energy = energy
            return True
        else:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

    def linker_translation_move(self):
        tries = 0
        
        while tries < 5:
            tries += 1
            
            first = system_random.choice(list(self.linker_start.keys()))
            first_coords = self.linker_space[first].coordinates
            second_coords = self.linker_space[first].neighbor

            axis = system_random.choice([0, 1, 2])
            direction = system_random.choice([1, -1])

            new_first = first_coords.copy()
            new_first[axis] += direction
            new_first = self.periodic_coordinate(new_first)

            if str(new_first) in self.linker_space:
                if str(new_first) != str(second_coords):
                    continue
            if str(new_first) in self.space and self.space[str(new_first)].polarity != 1:
                continue

            new_second = second_coords.copy()
            new_second[axis] += direction
            new_second = self.periodic_coordinate(new_second)

            if str(new_second) in self.linker_space:
                if str(new_second) != str(first_coords):
                    continue
            if str(new_second) in self.space and self.space[str(new_second)].polarity != -1:
                continue

            return self.move_chain([first_coords, second_coords], [new_first, new_second], None, 0, 'linker')

        if self.record_moves:
            self.acceptance_records.append(0)
            return False

    def linker_end_move(self):
        tries = 0

        while tries < 5:
            tries += 1
            first = random.choice(list(self.linker_start.keys()))
            first_coords = self.linker_space[first].coordinates
            second_coords = self.linker_space[first].neighbor
            
            start_or_end = system_random.choice([0,1])
            flip_coordinates = [first_coords, second_coords][start_or_end].copy()
            next_coordinates = flip_coordinates.copy()
            left_index = [i for i in [0,1] if i != start_or_end][0]
            back = [first_coords, second_coords][left_index]

            axis = None
            for i in range(3):
                if back[i] !=  next_coordinates[i]:
                    next_coordinates[i] = copy.copy(back[i])
                    axis = i
            next_axis = system_random.choice([i for i in [0,1,2] if i != axis])
            direction = system_random.choice([1, -1])
            next_coordinates[next_axis] += direction
            next_coordinates = self.periodic_coordinate(next_coordinates)

            if str(next_coordinates) not in self.linker_space:
                if str(next_coordinates) not in self.space or self.space[str(next_coordinates)].polarity == self.linker_space[str(flip_coordinates)].polarity:
                    return self.move_chain([flip_coordinates], [next_coordinates], None, start_or_end, 'linker')
                    
        if self.record_moves:
            self.acceptance_records.append(0)
            return False
            
    def simulate(self, n_mcmc=10000, interval=100, record_intervals=False, anneal=True, beta_lower_bound=0, beta_upper_bound=1, beta_interval=0.05):
        substep = round(n_mcmc*beta_interval/(beta_upper_bound - beta_lower_bound), 0)
        self.beta = beta_lower_bound - beta_interval
        self.energy_records.append(copy.copy(round(self.energy,2)))
        self.beta_records.append(copy.copy(self.beta))
        self.native_contacts_records.append(copy.copy(self.native_contacts))
        self.individual_ncs_records.append(copy.copy(self.individual_ncs))

        for step in range(1, n_mcmc+1):
            if anneal:
                if step%substep == 0:
                    self.beta += beta_interval
                    self.beta = round(self.beta, 2)
            all_functions = [self.end_move, self.corner_move, self.corner_move_anywhere, self.corner_flip, self.crankshaft_move, self.reptation_move, 
                            self.rotation_move]
            if self.n_polymers > 1:
                all_functions = all_functions + [self.transform_move]
                if self.allow_cluster_move:
                    all_functions = all_functions + [self.cluster_move]
            system_random.choice(all_functions)()
            self.validate_chain()
            all_linker_functions = [self.linker_translation_move, self.linker_end_move]
            for i in range(5):
                system_random.choice(all_linker_functions)()
                self.validate_chain()
            self.n_mcmc += 1
            if record_intervals and step%interval == 0 and step > 0:
                self.energy_records.append(copy.copy(round(self.energy,2)))
                self.beta_records.append(copy.copy(self.beta))
                self.native_contacts_records.append(copy.copy(self.native_contacts))
                self.individual_energy_records.append(copy.copy(self.individual_energy))
                self.individual_ncs_records.append(copy.copy(self.individual_ncs))

            if step%(n_mcmc/10) == 0:
                print('Completion: {}%'.format(step*100/n_mcmc))
                sys.stdout.flush()
            
            if self.record_moves and len(self.acceptance_records) != 6*self.n_mcmc:
                raise ValueError('Recorded acceptance records are not equal to the number of MCMC steps')
        print('Fully Completed and Validated.')

    def energy_variation_graph(self, savefigure=False, figure_name='LatticePy_energy.png'):
        n_total = len(self.energy_records)
        if n_total == 0:
            raise RuntimeError('No energy values have been recorded so far. Please rerun the simulation with record_intervals = True in the simulate() function.')
        
        interval = self.n_mcmc/n_total
        x = [interval*i for i in range(n_total)]
        plt.figure(figsize=(12,6))
        plt.plot(x, self.energy_records, linestyle='solid', linewidth=2, markersize=0)
        plt.title('Variation of system energy over all MCMC steps')
        plt.xlabel('Number of MCMC steps')
        plt.ylabel('System Energy')
        if savefigure:
            plt.savefig(figure_name, dpi=300, format='png')
        else:
            plt.show()
    
    def native_contacts_over_time(self, savefigure=False, figure_name='LatticePy_native_contacts.png'):
        n_total = len(self.native_contacts_records)
        interval = self.n_mcmc/n_total
        x = [interval*i for i in range(n_total)]
        plt.figure(figsize=(12,6))
        plt.plot(x, self.native_contacts_records, linestyle='solid', linewidth=2, markersize=0)
        plt.title('Variation of Native Contacts over all MCMC steps')
        plt.xlabel('Number of MCMC steps')
        plt.ylabel('Number of Native Contacts')
        if savefigure:
            plt.savefig(figure_name, dpi=300, format='png')
        else:
            plt.show()
    
    def nchc_over_time(self, savefigure=False, figure_name='LatticePy_nchc.png'):
        n_total = len(self.nchc_records)
        interval = self.n_mcmc/n_total
        x = [interval*i for i in range(n_total)]
        plt.figure(figsize=(12,6))
        plt.plot(x, self.nchc_records, linestyle='solid', linewidth=2, markersize=0)
        plt.title('Variation of NCHC Contacts over all MCMC steps')
        plt.xlabel('Number of MCMC steps')
        plt.ylabel('Number of Native Contacts')
        if savefigure:
            plt.savefig(figure_name, dpi=300, format='png')
        else:
            plt.show()

    def animate(self):
        def make_figure(i):
            df = self.records[i]
            return go.Scatter3d(x=df.x, y=df.y, z=df.z, mode='markers+lines',
                    marker = dict(size = 7, color = df.c, opacity = 0.8 ), showlegend=False )

        all_data = [[make_figure(i)] for i in range(len(self.records))]
        frames = []

        for i in range(len(all_data)):
            df = self.records[i]
            x_min = min(df.x)-2
            y_min = min(df.y)-2
            z_max = max(df.z)+2
            energy = self.energy_records[i]
            beta = self.beta_records[i]
            annotations = [dict(x=x_min, y=y_min, z=z_max, text='Energy: {}'.format(energy), showarrow=False),  dict(x=x_min, y=y_min, z=z_max+2, text='Current Beta: {}'.format(beta), showarrow=False) ]  
            layout = go.layout(scene=dict(annotations=annotations))
            frame = go.Frame(data=all_data[i], layout=layout)
            frames.append(frame)

        first_frame = self.records[0]

        fig = go.Figure(data=[go.Scatter3d(x=first_frame.x, y=first_frame.y, z=first_frame.z, 
                            mode='markers+lines', marker = dict(size = 7, color = first_frame.c, opacity = 0.8 ), showlegend=False )],

            layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]
            ), frames=frames)

        fig.show()
        
    def visualize(self, simulating=False, to_html=False, html_file_name='LatticePy_figure.html'):
        data = []
        x_min = None
        y_min = None
        z_max = None
        for end_coordinates in list(self.last.keys()):
            x = []
            y = []
            z = [] 
            polarities = []
            current = self.space[str(end_coordinates)]
            
            while True:
                coordinates = current.coordinates
                x.append(coordinates[0])
                y.append(coordinates[1])
                z.append(coordinates[2])
                polarities.append(current.polarity)
                if current.previous is None:
                    break
                else:
                    current = self.space[str(current.previous)]
            
            colors = {}
            colors[1] = 'red'
            colors[-1] = 'blue'
            df = pd.DataFrame()
            df['x'] = x
            df['y'] = y
            df['z'] = z
            df['c'] = [colors[i] for i in polarities]

            if simulating:
                return df
                
            fig1 = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers+lines', marker = dict(
                                         size = 7,
                                         color = df['c'],
                                         opacity = 0.8 ), showlegend=False)
            
            data.append(fig1)

        for start_coordinates in list(self.linker_start.keys()):
            x = []
            y = []
            z = [] 
            polarities = []
            current = self.linker_space[str(start_coordinates)]
            
            coordinates = current.coordinates
            x.append(coordinates[0])
            y.append(coordinates[1])
            z.append(coordinates[2])
            polarities.append(current.polarity)

            neighbor = self.linker_space[str(current.neighbor)]
            coordinates = neighbor.coordinates
            x.append(coordinates[0])
            y.append(coordinates[1])
            z.append(coordinates[2])
            polarities.append(neighbor.polarity)
            
            colors = {}
            colors[1] = 'green'
            colors[-1] = 'black'
            df = pd.DataFrame()
            df['x'] = x
            df['y'] = y
            df['z'] = z
            df['c'] = [colors[i] for i in polarities]

            if simulating:
                return df
    
            fig1 = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers+lines', marker = dict(
                                         size = 8,
                                         color = df['c'],
                                         opacity = 0.8 ), showlegend=False)
            
            data.append(fig1)
         
        fig = go.Figure(data=data)

        if to_html:
            fig.write_html(html_file_name)
        else:
            plotly.offline.iplot(fig, filename='simple-3d-scatter')
