import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
import pandas as pd
import numpy as np
from .objects import amino_acid
from random import randint, choice, random
import plotly.graph_objects as go
import plotly
import plotly.io as pio
pio.renderers.default = 'iframe_connected'
import sys
import copy
import networkx as nx

class lattice():
    def __init__(self, bound, E_c, beta=0, lattice_type='simple_cubic'):
        self.lattice_type = lattice_type
        self.bound = bound
        self.space = dict()
        self.start = dict()
        self.last = dict()
        self.energy = 0
        self.all_bonds = []
        self.bond_energies = dict()
        self.bond_energies[1] = {}
        self.bond_energies[-1] = {}
        self.beta = beta
        self.length_of_polymer = 0
        self.energy_records = []
        self.beta_records = []
        self.records = []
        self.current_move = None
        self.native_contacts = 0
        self.non_covalent_hydrophobic_contacts = 0
        self.n_mcmc = 0
        
        bonds = [-1,-1,1,-1,1,1]
        bond_energies = [-2.3-E_c, -1-E_c, -E_c]
        for i in range(3):
            self.bond_energies[bonds[2*i]][bonds[2*i+1]] = bond_energies[i]
            self.bond_energies[bonds[2*i+1]][bonds[2*i]] = bond_energies[i]
        
        for i in range(self.bound+1):
            for j in range(self.bound+1):
                for k in range(self.bound+1):
                    self.space[str([i, j, k])] = amino_acid(0, [i, j, k])
                    self.space[str([-i, -j, -k])] = amino_acid(0, [-i, -j, -k])
                    self.space[str([-i, -j, k])] = amino_acid(0, [-i, -j, k])
                    self.space[str([-i, j, -k])] = amino_acid(0, [-i, j, -k])
                    self.space[str([i, -j, -k])] = amino_acid(0, [i, -j, -k])
                    self.space[str([-i, j, k])] = amino_acid(0, [-i, j, k])
                    self.space[str([i, -j, k])] = amino_acid(0, [i, -j, k])
                    self.space[str([i, j, -k])] = amino_acid(0, [i, j, -k])  

    def update_system_energy(self, nearest_neighbors=True):
        original_connections=nx.Graph()
        energy = 0
        hydrophobic_contacts = 0
        for last in list(self.last.keys()):
            aa = self.space[str(last)]
            while aa.polarity != 0:
                if nearest_neighbors:
                    neighbors = []
                    for i in range(3):
                        neighbor = aa.coordinates.copy()
                        neighbor[i] += 1
                        neighbor_rev = aa.coordinates.copy()
                        neighbor_rev[i] += -1
                        neighbors.append(neighbor)
                        neighbors.append(neighbor_rev)
                    neighbors = [neighbor for neighbor in neighbors if self.space[str(neighbor)].polarity != 0 and neighbor not in [aa.next, aa.previous] and \
                                        (str(aa.coordinates), str(neighbor)) not in original_connections.edges ]
                    for neighbor in neighbors:
                        energy += self.bond_energies[aa.polarity][self.space[str(neighbor)].polarity]
                        original_connections.add_edge(str(aa.coordinates), str(neighbor))
                        if aa.polarity == -1 and self.space[str(neighbor)].polarity == -1:
                            hydrophobic_contacts += 1
                if aa.previous != None:
                    next = self.space[str(aa.previous)]
                    aa=next
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
        for aa in originals:
            neighbors = []
            for i in range(3):
                neighbor_int = [int(i) for i in aa.strip('][').split(', ')].copy()
                neighbor = neighbor_int.copy()
                neighbor_rev = neighbor_int.copy()
                neighbor[i] += 1
                neighbor_rev[i] += -1
                neighbors.append(neighbor)
                neighbors.append(neighbor_rev)

            if i == 0:
                if start_or_end == 0:
                    exclude = originals + [self.space[aa]].next 
                else:
                    exclude = originals + [self.space[aa]].previous
            else:
                exclude = originals

            neighbors = [neighbor for neighbor in neighbors if self.space[str(neighbor)].polarity != 0 and str(neighbor) not in exclude \
                                     and (str(aa), str(neighbor)) not in original_connections.edges]
            for neighbor in neighbors:
                original_energy += self.bond_energies[self.space[aa].polarity][self.space[str(neighbor)].polarity]
                original_connections.add_edge(str(aa),str(neighbor))
                if self.space[aa].polarity == -1 and self.space[str(neighbor)].polarity == -1:
                    original_NCHC += 1
            i += 1

        new_connections=nx.Graph()
        new_NCHC = 0
        new_energy = 0
        i = 0
        for (original, aa) in zip(originals, replacements):
            neighbors = []
            for i in range(3):
                neighbor_int = [int(i) for i in aa.strip('][').split(', ')].copy()
                neighbor = neighbor_int.copy()
                neighbor_rev = neighbor_int.copy()
                neighbor[i] += 1
                neighbor_rev[i] += -1
                neighbors.append(neighbor)
                neighbors.append(neighbor_rev)

            if i == 0:
                if start_or_end == 0:
                    exclude = [self.space[str(originals[0])].next] + non_existing + replacements
                else:
                    exclude = [self.space[str(originals[0])].previous] + non_existing + replacements
            else:
                exclude = non_existing + replacements

            neighbors = [neighbor for neighbor in neighbors if str(neighbor) not in exclude and (str(aa), str(neighbor)) not in new_connections.edges \
                                and self.space[str(neighbor)].polarity != 0]
            for neighbor in neighbors:
                new_energy += self.bond_energies[self.space[original].polarity][self.space[str(neighbor)].polarity]
                new_connections.add_edge(str(aa),str(neighbor))
                if self.space[original].polarity == -1 and self.space[str(neighbor)].polarity == -1:
                    new_NCHC += 1
            i += 1
        return new_energy - original_energy, len(new_connections.edges) - len(original_connections.edges), new_NCHC - original_NCHC

    def move_success(self, deltaE):
        if random() < np.exp(-deltaE*self.beta):
            return True
        else:
            return False

    def add_protein(self, sequence=None, type='straight', n_polymers=1):
        polymer = []

        for aa in list(sequence):
            if aa in 'GAVCPLIMWFKRH':
                polymer.append(+1)
            elif aa in 'STYNQDE':
                polymer.append(-1)
            else:
                raise InputError('Unrecognized amino acid in sequence: {}'.format(aa))

        if type == 'straight':
            self.add_polymer_straight(polymer, n_polymers=n_polymers)
        elif type == 'random':
            self.add_polymer_randomly(polymer, n_polymers=n_polymers)
        else:
            raise InputError('Unrecognized type of polymer placement: {}'.format(type))
            
    def add_polymer_straight(self, polymer, n_polymers=1):
        length = len(polymer)
        x = randint(-self.bound+length+1, self.bound-length-1)
        y = randint(-self.bound+length+1, self.bound-length-1)
        z = randint(-self.bound+length+1, self.bound-length-1)
        polymers_placed = 0
        axis = 0
        direction = 0

        while polymers_placed < n_polymers:
            new_coords = [x,y,z]
            polymer_id = polymers_placed
            valid_path=False

            while polymers_placed>0 and valid_path is False:
                new_coords = choice(list(self.start.keys()))
                new_coords = [int(i) for i in new_coords.strip('][').split(', ')].copy()
                axis_start = choice([0,1,2])
                change = choice([1,2])
                new_coords[axis_start] += choice([+change, -change])
                if str(new_coords) not in self.start.keys() and self.space[str(new_coords)].polarity == 0:
                    all_polarities = 0
                    start = new_coords.copy()
                    for i in range(length):
                        new = start.copy()
                        new[axis] += direction*i
                        if str(new) not in self.space:
                            valid_path = False
                            break
                        else:
                            all_polarities += self.space[str(new)].polarity
                        if i == length - 1 and all_polarities==0:
                            valid_path=True

            if polymers_placed == 0:
                valid_path=False
                while valid_path!= True:
                    axis = choice([0,1,2])
                    direction = choice([1, -1])
                    all_polarities = 0
                    start = new_coords.copy()
                    for i in range(length):
                        new = start.copy()
                        new[axis] += direction*i
                        if str(new) not in self.space:
                            valid_path = False
                            break
                        else:
                            all_polarities += self.space[str(new)].polarity
                        if i == length - 1 and all_polarities==0:
                            valid_path=True

            self.start[str(new_coords)] = polymer_id
            self.space[str(new_coords)] = amino_acid(polymer[0],new_coords, polymer_id)
            current_aa = self.space[str(new_coords)]
        
            for i in range(1, length):
                next_coordinates = current_aa.coordinates.copy()
                next_coordinates[axis] += direction
                self.space[str(current_aa.coordinates)].next = next_coordinates
                self.space[str(next_coordinates)] = amino_acid(polymer[i], next_coordinates, polymer_id)
                self.space[str(next_coordinates)].previous = current_aa.coordinates.copy()
                current_aa = self.space[str(next_coordinates)]
            polymers_placed += 1
            self.length_of_polymer = length
            self.last[str(next_coordinates)] = 1
            self.update_system_energy()

    def add_polymer_randomly(self, polymer):
        length = len(polymer)
        polymer_id = len(self.last.keys())
        x = randint(-self.bound+length, self.bound-length)
        y = randint(-self.bound+length, self.bound-length)
        z = randint(-self.bound+length, self.bound-length)
        
        while True:
            if self.space[str([x, y, z])].polarity == 0:
                self.start[str([x, y, z])] = 1
                break
            else:
                x = randint(-self.bound+length, self.bound-length)
                y = randint(-self.bound+length, self.bound-length)
                z = randint(-self.bound+length, self.bound-length)

        self.space[str([x, y, z])] = amino_acid(polymer[0], [x, y, z], polymer_id)
        current_a = self.space[str([x, y, z])]

        for i in range(1, length):
            coordinates = current_a.coordinates
            while True:
                next_coordinates = coordinates.copy()
                axis = randint(0,2)
                next_coordinates[axis] += randint(-1,1)
                if self.space[str(next_coordinates)].polarity == 0:
                    break
            self.space[str(coordinates)].next = next_coordinates
            self.space[str(next_coordinates)] = amino_acid(polymer[i], next_coordinates, polymer_id)
            self.space[str(next_coordinates)].previous = coordinates
            current_a = self.space[str(next_coordinates)]

        self.last[str(next_coordinates)] = 1
        self.update_system_energy()

    def move_chain(self, originals, replacements, inflection_point, start_or_end):
        deltaE, deltaNC, deltaNCHC = self.find_subsystem_energy(originals, replacements, start_or_end)
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
                        #original_aa.next = inflection_point.copy()#else:
                    if original_aa.previous is not None and len(replacements) > 1:
                        original_aa.previous = [int(i) for i in replacements[aa_step+1].strip('][').split(', ')].copy()
                elif start_or_end == 1:
                    if aa_step != 0:
                        original_aa.previous = [int(i) for i in replacements[aa_step-1].strip('][').split(', ')].copy()
                        #original_aa.previous = inflection_point.copy()#else:
                    if original_aa.next is not None and len(replacements) > 1:
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
                    self.space[original] = amino_acid(0, original_int)

    def validate_chain(self):
        i = 0
        start = self.space[list(self.start.keys())[0]]
        records = []
        while i < 27:
            if start.polarity == 0:
                print(i)
                print(start.coordinates)
                sys.exit('polarity problem')
            if start.coordinates is None:
                print(i)
                sys.exit('coordinates are none')
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
            start = self.space[str(next_coordinate)]
        return True

    def end_move(self):
        self.validate_chain()
        start_or_end=choice([0,1])
        deltaE = 0
        coordinates = choice([ list(self.start.keys()), list(self.last.keys())][start_or_end])
        coordinates = coordinates.strip('][').split(', ')
        coordinates = [int(i) for i in coordinates]
        if abs(coordinates[0]) >= self.bound-2 or abs(coordinates[1]) >= self.bound-2 or abs(coordinates[2])  >= self.bound-2:
            return False
        current_aa = copy.copy(self.space[str(coordinates)])
        current_polarity = current_aa.polarity
        next_aa_polarity = copy.copy(self.space[str(coordinates)].polarity)
        next_coordinates = None
        tries = 0
        back = []

        while next_aa_polarity != 0 and tries < 5:
            tries += 1
            next_coordinates = coordinates.copy()  
            back = None
            if current_aa.next == None:
                back = current_aa.previous.copy()
            else:
                back = current_aa.next.copy()
            axis = None
            for i in range(3):
                if back[i] !=  next_coordinates[i]:
                    next_coordinates[i] = copy.copy(back[i])
                    axis = i
            next_axis = choice([i for i in [0,1,2] if i != axis])
            direction = choice([1, -1])
            next_coordinates[next_axis] += direction

            if str(next_coordinates) not in self.space:
                continue 

        if self.space[str(next_coordinates)].polarity != 0:
            return False
        
        self.validate_chain()
        self.move_chain([str(coordinates)], [str(next_coordinates)], back, start_or_end)
        self.validate_chain()

    def corner_move(self):
        self.validate_chain()
        polymer_id=choice(range(len(self.start.keys())))
        start_or_end=choice([0,1])
        deltaE = 0
        coordinates = [ list(self.start.keys()), list(self.last.keys())][start_or_end][polymer_id]
        coordinates = coordinates.strip('][').split(', ')
        coordinates = [int(i) for i in coordinates]
        if abs(coordinates[0]) >= self.bound-2 or abs(coordinates[1]) >= self.bound-2 or abs(coordinates[2])  >= self.bound-2:
            return False
        current_aa = self.space[str(coordinates)]
        if start_or_end == 0:
            next_coordinates = current_aa.next.copy()
        else:
            next_coordinates = current_aa.previous.copy()
        first_rep = coordinates.copy()
        second_rep = next_coordinates.copy()
        tries = 0
        while True and tries < 5:
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
            next_axis = choice([i for i in [0,1,2] if i != axis])
            next_direction = choice([1, -1])
            second_rep[next_axis] += next_direction
            first_rep = coordinates.copy()
            first_rep[axis] += first_direction
            first_rep[next_axis] += next_direction
            if self.space[str(first_rep)].polarity == 0 and self.space[str(second_rep)].polarity == 0:
                break
        if self.space[str(first_rep)].polarity != 0 or self.space[str(second_rep)].polarity != 0:
            return False
        originals = [str(next_coordinates), str(coordinates)]
        replacements = [str(second_rep), str(first_rep)]
        self.move_chain(originals, replacements, third, start_or_end)
        self.validate_chain()

    def corner_move_anywhere(self):
        self.validate_chain()
        polymer_id=choice(range(len(self.start.keys())))
        start_or_end=choice([0,1])
        coordinates = [ list(self.start.keys()), list(self.last.keys())][start_or_end][polymer_id]
        coordinates = coordinates.strip('][').split(', ')
        coordinates = [int(i) for i in coordinates]

        if abs(coordinates[0]) >= self.bound-5 or abs(coordinates[1]) >= self.bound-5 or abs(coordinates[2])  >= self.bound-5:
            return False

        steps = randint(2, self.length_of_polymer-2)
        all_choices = []

        if start_or_end == 0:
            for step in range(steps):
                coordinates = self.space[str(coordinates)].next.copy()
                all_choices.append(coordinates)
        else:
            for step in range(steps):
                coordinates = self.space[str(coordinates)].previous.copy()
                all_choices.append(coordinates)

        inflection_point = choice(all_choices)
        valid_path = False
        tries = 0
        to_be_replaced = {}
        replacements = {}
        
        while not valid_path and tries < 5:
            tries += 1
            if start_or_end == 0:
                first_aa = self.space[str(inflection_point)].previous.copy()
                next_aa = copy.deepcopy(self.space[str(self.space[str(inflection_point)].previous)])
            else:
                first_aa = self.space[str(inflection_point)].next.copy()
                next_aa = copy.deepcopy(self.space[str(self.space[str(inflection_point)].next)])

            first_axis = None
            first_direction = 0
            for i in range(3):
                if first_aa[i] != copy.copy(inflection_point[i]):
                    first_direction = inflection_point[i] - first_aa[i]
                    first_aa[i] = copy.copy(inflection_point[i])
                    first_axis = i
            next_axis = choice([i for i in [0,1,2] if i != first_axis])
            next_direction = choice([1, -1])
            first_aa[next_axis] += next_direction
            to_be_replaced = {}
            to_be_replaced_list = []
            replacements = {}

            while True:
                to_be_replaced[str(next_aa.coordinates.copy())] = 1
                to_be_replaced_list.append(next_aa.coordinates.copy())

                if start_or_end == 0:
                    if next_aa.previous is None:
                        break
                    next_aa = self.space[str(next_aa.previous)]
                else:
                    if next_aa.next is None:
                        break
                    next_aa = self.space[str(next_aa.next)]

            valid_path = True
            i = 0
            while valid_path and i < len(to_be_replaced.keys()):
                original = to_be_replaced_list[i].copy()
                new_position = to_be_replaced_list[i].copy()
                new_position[first_axis] += first_direction
                new_position[next_axis] += next_direction
                next_aa = self.space[str(new_position)]
                if next_aa.polarity != 0 and str(new_position) not in to_be_replaced:
                    valid_path = False
                else:
                    replacements[str(new_position.copy())] = 1
                    i += 1

        if not valid_path:
            return False

        if len(to_be_replaced_list) < 3:
            return False

        if len(to_be_replaced.keys()) == len(replacements.keys()) and len(to_be_replaced.keys())>0:
            self.validate_chain()
            self.move_chain( list(to_be_replaced.keys()), list(replacements.keys()), inflection_point, start_or_end)
            self.validate_chain()

    def corner_flip(self):
        done = False
        tries = 5
        start=choice(list(self.start.keys()))
        center_coordinates = self.space[start].next.copy()
        replacement = None

        while not done and center_coordinates is not None and tries < 5:
            tries += 5
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

            if before_axis == next_axis:
                center_coordinates = self.space[str(center_coordinates)].next
                continue
            else:
                replacement = center_coordinates.copy()
                replacement[before_axis] += first_direction
                replacement[next_axis] += next_direction
                if self.space[str(replacement)].polarity == 0:
                    self.move_chain([str(center_coordinates)], [str(center_coordinates)], before, 0)
                    done = True
                else:
                    center_coordinates = self.space[str(center_coordinates)].next

    def simulate(self, n_mcmc=10000, interval=100, record_intervals=False, anneal=True, beta_lower_bound=0, beta_upper_bound=1, beta_interval=0.05):
        substep = round(n_mcmc*beta_interval/(beta_upper_bound - beta_lower_bound), 0)
        self.beta = beta_lower_bound - beta_interval
        self.n_mcmc += n_mcmc

        for step in range(n_mcmc):

            if anneal:
                if step%substep == 0:
                    self.beta += beta_interval
                    self.beta = round(self.beta, 2)

            num = random()
            if num < 0.25:
                self.current_move = 'end_move'
                self.end_move()
                
            elif num >= 0.25 and num < 0.5:
                self.current_move = 'corner_move'
                self.corner_move()
            elif num >= 0.5 and num < 0.75:
                self.current_move = 'corner_anywhere'
                self.corner_move_anywhere()
            else:
                self.current_move = 'corner_flip'
                self.corner_flip()

            if record_intervals and step%interval == 0 and step > 0:
                out = self.visualize(simulating=True)
                self.records.append(out)
                self.energy_records.append(copy.copy(self.energy))
                self.beta_records.append(copy.copy(self.beta))

            if step%(n_mcmc/10) == 0:
                print('Completion: {}%'.format(step*100/n_mcmc))
                sys.stdout.flush()

        print('Fully Completed and Validated.')

    def energy_variation_graph(self):
        n_total = len(self.energy_records)
        interval = self.n_mcmc/n_total

        x = [interval*i for i in range(n_total)]

        plt.figure(figsize=(12,6))
        plt.plot(x, self.energy_records, linestyle='solid', linewidth=2, markersize=0)
        plt.title('Variation of system energy over all MCMC steps')
        plt.xlabel('Number of MCMC steps')
        plt.ylabel('System Energy')
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
        
    def visualize(self, simulating=False):
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
        plotly.offline.iplot(fig, filename='simple-3d-scatter')