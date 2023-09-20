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
from .objects import amino_acid
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sns.set()
system_random = random.SystemRandom()
pio.renderers.default = 'iframe_connected'

class lattice():
    def __init__(self, bound, E_c, beta=0, lattice_type='simple_cubic', record_moves=False):
        self.lattice_type = lattice_type
        self.record_moves = record_moves
        self.bound = bound
        self.space = dict()
        self.start = dict()
        self.last = dict()
        self.energy = 0
        self.bond_energies = dict()
        self.bond_energies[1] = {}
        self.bond_energies[-1] = {}
        self.native_contacts = 0
        self.non_covalent_hydrophobic_contacts = 0
        self.beta = beta
        self.n_mcmc = 0
        self.length_of_polymer = 0
        self.energy_records = []
        self.beta_records = []
        self.native_contacts_records = []
        self.nchc_records = []
        self.move_records = []
        self.acceptance_records = []
        self.records = []
        self.n_polymers = 0
        self.plateau_time = 0
        self.plateaued = False
        bonds = [-1,-1,1,-1,1,1]
        bond_energies = [-2.3-E_c, -1-E_c, -E_c]
        
        for i in range(3):
            self.bond_energies[bonds[2*i]][bonds[2*i+1]] = bond_energies[i]
            self.bond_energies[bonds[2*i+1]][bonds[2*i]] = bond_energies[i]
            
    def periodic_coordinate(self, coordinates):
        return [coordinates[0]%31, coordinates[1]%31, coordinates[2]%31]
                
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
            neighbors = [neighbor for neighbor in neighbors if str(neighbor) in self.space and str(neighbor) not in [str(aa.next),str(aa.previous)]]
            return neighbors
    
    def update_system_energy(self, nearest_neighbors=True):
        original_connections=nx.Graph()
        energy = 0
        hydrophobic_contacts = 0
        for last in list(self.last.keys()):
            aa = self.space[str(last)]
            while True:
                if nearest_neighbors:
                    neighbors = self.find_valid_neighbors(aa)
                    for neighbor in neighbors:
                        energy += self.bond_energies[aa.polarity][self.space[str(neighbor)].polarity]
                        original_connections.add_edge(str(aa.coordinates), str(neighbor))
                        if aa.polarity == -1 and self.space[str(neighbor)].polarity == -1:
                            hydrophobic_contacts += 1
                if aa.previous != None:
                    next_aa = self.space[str(aa.previous)]
                    aa=next_aa
                else:
                    break
        self.native_contacts = len(original_connections.edges)
        self.non_covalent_hydrophobic_contacts = hydrophobic_contacts
        self.energy = energy

    def find_subsystem_energy(self, originals, replacements, start_or_end):
        original_energy = 0
        non_existing = [i for i in originals if i not in replacements]
        original_connections=nx.Graph()
        original_NCHC = 0
        i  = 0
        for coord in originals:
            aa = self.space[coord]
            neighbors = self.find_valid_neighbors(aa)
            neighbors = [neighbor for neighbor in neighbors if str(neighbor) not in originals and (coord, str(neighbor)) not in original_connections.edges ]
            for neighbor in neighbors:
                original_energy += self.bond_energies[self.space[str(aa.coordinates)].polarity][self.space[str(neighbor)].polarity]
                original_connections.add_edge(str(aa.coordinates),str(neighbor))
                if self.space[str(aa.coordinates)].polarity == -1 and self.space[str(neighbor)].polarity == -1:
                    original_NCHC += 1
            i += 1

        new_connections=nx.Graph()
        new_NCHC = 0
        new_energy = 0
        i = 0
        for (original, aa) in zip(originals, replacements):
            new_object = amino_acid(0, [int(i) for i in aa.strip('][').split(', ')].copy() , None)
            original_aa = self.space[original]
            neighbors = self.find_valid_neighbors(new_object)
            exclude = non_existing + replacements + [str(original_aa.previous), str(original_aa.next)]
            neighbors = [neighbor for neighbor in neighbors if str(neighbor) not in exclude and (aa, str(neighbor)) not in new_connections.edges]
            for neighbor in neighbors:
                new_energy += self.bond_energies[self.space[original].polarity][self.space[str(neighbor)].polarity]
                new_connections.add_edge(aa,str(neighbor))
                if self.space[original].polarity == -1 and self.space[str(neighbor)].polarity == -1:
                    new_NCHC += 1
            i += 1
        return new_energy - original_energy, len(new_connections.edges) - len(original_connections.edges), new_NCHC - original_NCHC

    def move_success(self, deltaE):
        if system_random.random() < np.exp(-deltaE*self.beta):
            return True
        else:
            return False

    def add_protein(self, sequence=None, placement='straight', n_polymers=1):
        polymer = []

        for aa in list(sequence):
            if aa in 'GAVCPLIMWFKRH':
                polymer.append(+1)
            elif aa in 'STYNQDE':
                polymer.append(-1)
            else:
                raise InputError('Unrecognized amino acid in sequence: {}'.format(aa)) # noqa: F821

        if placement in ['straight','random']:
            self.add_polymer(polymer, n_polymers=n_polymers, placement=placement)
        else:
            raise InputError('Unrecognized type of polymer placement: {}'.format(type)) # noqa: F821
            
    def add_polymer(self, polymer, n_polymers=1, placement='straight'):
        length = len(polymer)
        polymers_placed = 0

        while polymers_placed < n_polymers:
            polymer_id = polymers_placed
            valid_path=False
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
            valid_path=True
            
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
                self.start[str(new_coords)] = len(self.start)
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
                self.last[str(next_coordinates)] = 1
                self.update_system_energy()
                self.n_polymers += 1

    def move_chain(self, originals, replacements, inflection_point, start_or_end):
        deltaE, deltaNC, deltaNCHC = self.find_subsystem_energy(originals, replacements, start_or_end)
        self.validate_chain()
        if self.move_success(deltaE):
            self.energy += deltaE
            self.native_contacts += deltaNC
            self.non_covalent_hydrophobic_contacts += deltaNCHC
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
                
            for aa_step in range(len(originals)):
                original = originals[aa_step]
                original_int = [int(i) for i in original.strip('][').split(', ')].copy()
                replacement = replacements[aa_step]
                replacement_int = [int(i) for i in replacement.strip('][').split(', ')].copy()
                replacement_object = copy.copy(all_objects[aa_step])
                if aa_step ==0:
                    if start_or_end == 0:
                        self.space[str(inflection_point)].previous = replacement_int.copy()
                    else:
                        self.space[str(inflection_point)].next = replacement_int.copy()
                if aa_step == len(originals)-1:
                    if original in self.last.keys():
                        del self.last[copy.copy(original)]
                        self.last[replacement] = 1
                    elif original in self.start.keys():
                        del self.start[copy.copy(original)]
                        self.start[replacement] = 1
                    if start_or_end == 0:
                        if replacement_object.previous is not None:
                            self.space[str(replacement_object.previous)].next = replacement_int
                    else:
                        if replacement_object.next is not None:
                            self.space[str(replacement_object.next)].previous = replacement_int
                self.space[replacement] = replacement_object
                if original not in replacements:
                    del self.space[original]
            self.validate_chain()
            if self.record_moves:
                self.acceptance_records.append(1)
            return True
        else:
            if self.record_moves:
                self.acceptance_records.append(0)
            return False

    def validate_chain(self):
        for start in list(self.start.keys()):
            i = 0
            records = []
            start = self.space[start]
            while i < 27:
                if max(start.coordinates) > 30 or min(start.coordinates) < 0:
                    raise RuntimeError('Periodic Boundary Conditions have been violated.')
                records.append(start.coordinates)
                i+= 1
                next_coordinate = copy.copy(start.next)
                if next_coordinate is None:
                    if i==27:
                        break
                    else:
                        print(i)
                        print(start.coordinates)
                        sys.exit('broken chain in validation')
                if next_coordinate in records:
                    print('\n')
                    print(records)
                    print(next_coordinate)
                    sys.exit('chain duplicated')

                if math.dist(start.coordinates, start.next) not in [1, 30]:
                    print(start.coordinates, start.next)
                    sys.exit('invalid neighbors')

                start = self.space[str(next_coordinate)]
        return True

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
                return self.move_chain([str(coordinates)], [str(next_coordinates)], back, start_or_end)
        
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
                        return self.move_chain([str(second_coordinates), str(first_coordinates)], [str(second_try), str(first_try)], third_coordinates, 0)
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
                return self.move_chain(originals, replacements, third, start_or_end)
            
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
            return self.move_chain( list(to_be_replaced.keys()), list(replacements.keys()), inflection_point, start_or_end)
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
                    return self.move_chain([str(center_coordinates)], [str(replacement)], after, 0)
            center_coordinates = self.space[str(center_coordinates)].next
        if self.record_moves:
            self.acceptance_records.append(0)
        
    def reptation_move(self):
        return True

    def simulate(self, n_mcmc=10000, interval=100, record_intervals=False, anneal=True, beta_lower_bound=0, beta_upper_bound=1, beta_interval=0.05):
        substep = round(n_mcmc*beta_interval/(beta_upper_bound - beta_lower_bound), 0)
        self.beta = beta_lower_bound - beta_interval

        for step in range(n_mcmc):
            if anneal:
                if step%substep == 0:
                    self.beta += beta_interval
                    self.beta = round(self.beta, 2)
            all_functions = [self.end_move, self.corner_move, self.corner_move_anywhere, self.corner_flip, self.crankshaft_move]
            if self.n_polymers > 1:
                all_functions.append(self.reptation_move)
            system_random.choice(all_functions)()
            self.n_mcmc += 1
            if record_intervals and step%interval == 0 and step > 0:
                out = self.visualize(simulating=True)
                self.records.append(out)
                self.energy_records.append(copy.copy(self.energy))
                self.beta_records.append(copy.copy(self.beta))
                self.native_contacts_records.append(copy.copy(self.native_contacts))
                self.nchc_records.append(copy.copy(self.non_covalent_hydrophobic_contacts))
                if not self.plateaued and len(self.energy_records) > 30 and np.var(self.energy_records[-30:-1]) < 2:
                    self.plateau_time = copy.copy(self.n_mcmc)
                    self.plateaued = True
            if step%(n_mcmc/10) == 0:
                print('Completion: {}%'.format(step*100/n_mcmc))
                sys.stdout.flush()
            
            if self.record_moves and len(self.acceptance_records) != self.n_mcmc:
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
                if current.previous == None:
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
            if x_min is None:
                x_min = min(df.x)
                y_min = min(df.y)
                z_max = min(df.z)

            if min(df.x) < x_min:
                x_min = min(df.x)
            if min(df.y) < y_min:
                y_min = min(df.y)
            if max(df.z) > z_max:
                z_max = max(df.z)
    
            fig1 = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers+lines', marker = dict(
                                         size = 7,
                                         color = df['c'],
                                         opacity = 0.8 ), showlegend=False)
            
            data.append(fig1)

        annotations = [dict(x=x_min-2, y=y_min-2, z=z_max+1, text='Energy: {}'.format(round(self.energy,2)), showarrow=False), 
                       dict(x=x_min-2, y=y_min-2, z=z_max+3, text='Native Contacts: {}'.format(self.native_contacts), showarrow=False),
                       dict(x=x_min-2, y=y_min-2, z=z_max+5, text='Non Covalent Hydrophobic Contacts: {}'.format(self.non_covalent_hydrophobic_contacts), showarrow=False),
                       dict(x=x_min-2, y=y_min-2, z=z_max+7, text='Current Beta: {}'.format(round(self.beta,2)), showarrow=False)]          
        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(annotations=annotations))
        if to_html:
            fig.write_html(html_file_name)
        else:
            plotly.offline.iplot(fig, filename='simple-3d-scatter')
