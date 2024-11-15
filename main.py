import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys 
import math
import random
from tqdm import tqdm

#set recursion limit
sys.setrecursionlimit(10**6)

class defaultdict(dict):
    def __init__(self, default_factory=None):
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if self.default_factory is None:
                raise
            value = self.default_factory()
            self[key] = value
            return value

class PinCluster:
    def __init__(self, pins=[]):
        self.pins = pins if pins else []

    def add_pin(self, pin):
        self.pins.append(pin)

    def calculate_wire_length(self):
        # Calculate wire length for this cluster
        min_x = min(pin.x for pin in self.pins)
        max_x = max(pin.x for pin in self.pins)
        min_y = min(pin.y for pin in self.pins)
        max_y = max(pin.y for pin in self.pins)

        width = max_x - min_x
        height = max_y - min_y
        semi_perimeter = width + height

        return semi_perimeter
    
    
class Pin:
    def __init__(self,name,x,y,component, cluster = PinCluster(), prev = None, nxt = []):
        self.name = name
        self.x = x
        self.y = y
        self.cluster = cluster
        self.prev = prev
        self.nxt = nxt
        self.gate = component
        
    def my_out(self):
        return self.cluster.pins[-1]

class CircuitComponent:
    def __init__(self, name, width, height, delay, pins, x=0, y=0):
        self.name = name
        self.width = width
        self.height = height
        self.delay = delay
        self.pins = pins  # List of tuples [(name, x1, y1), (name, x2, y2), ...] for pin locations
        self.x = x  # x coordinate of bottom-left corner
        self.y = y  # y coordinate of bottom-left corner

    def get_pins(self):
        # Adjust pin coordinates based on gate's position (x, y)
        return self.pins
    
    # def update(self, w , h):
    #     self.x += w
    #     self.y += h
    #     for i in self.pins:
    #         i.x += w
    #         i.y += h

# class CircuitComponent:
#     def __init__(self, name, width, height, pins):
#         self.name = name
#         self.width = width
#         self.height = height
#         self.pins = pins
#         self.x = 0
#         self.y = 0

    def reposition(self, new_x, new_y):
        w = new_x - self.x
        h = new_y - self.y
        for i in self.pins:
            i.x += w
            i.y += h
        self.x = new_x
        self.y = new_y

    def get_pin_coordinates(self, pin_index):
        pin_x, pin_y = self.pins[pin_index]
        return self.x + pin_x, self.y + pin_y

    def get_footprint(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def is_colliding(self, other):
        x1, y1, x2, y2 = self.get_footprint()
        ox1, oy1, ox2, oy2 = other.get_footprint()
        return not (x1 >= ox2 or x2 <= ox1 or y1 >= oy2 or y2 <= oy1)
        


def is_output_pin(pin, component):
    """Check if a pin is an output pin (on the right boundary of the gate)."""
    return pin.x == component.x + component.width

def is_input_pin(pin, component):
    """Check if a pin is an input pin (on the left boundary of the gate)."""
    return pin.x == component.x



def parse_input_file(input_file_path, output_file_path):
    components_list = []
    wires = []
    components_dict = {}
    wire_delay = 0
    component_locations = {}

    # Step 1: Parse input.txt to get gate details, pins, and wires
    with open(input_file_path, 'r') as f_in:
        lines = f_in.readlines()
        current_component = None  # Store the current component

        for line in lines:
            line = line.strip()

            if line.startswith('g'):  # Parse gate details
                parts = line.split()
                name = parts[0]
                width = int(parts[1])
                height = int(parts[2])
                delay = int(parts[3])

                # Create a new CircuitComponent with initial location (0, 0)
                current_component = CircuitComponent(name, width, height, delay, [])
                components_list.append(current_component)
                components_dict[name] = current_component

            elif line.startswith('wire_delay'):  # Parse wire delay
                wire_delay = int(line.split()[1])

            elif line.startswith('pins'):  # Parse pin coordinates
                # Expected format: pins (name1 x1 y1) (name2 x2 y2) ...
                parts = line.split()[2:]  # Skip the 'pins' keyword and start from coordinates
                pins = []
                
                for i in range(0, len(parts), 2):
                    pin_name = f"{current_component.name}.{i // 2 + 1}"
                    pin_x = int(parts[i])
                    pin_y = int(parts[i + 1])

                    # Create Pin object and associate it with the current component
                    pin = Pin(pin_name, pin_x, pin_y, current_component)
                    pins.append(pin)

                current_component.pins = pins  # Assign parsed pins to the current component

            elif line.startswith('wire'):  # Parse wire connections
                # Expected format: wire g1.p1 g2.p2 (e.g., wire g1.p1 g2.p2)
                parts = line.split()
                gate1_name, pin1_name = parts[1].split('.')
                gate2_name, pin2_name = parts[2].split('.')

                # Find the components_list corresponding to the gates
                gate1 = next(c for c in components_list if c.name == gate1_name)
                gate2 = next(c for c in components_list if c.name == gate2_name)

                # Find the correct pins based on the pin names (e.g., p1 -> 0, p2 -> 1)
                pin1_index = int(pin1_name[1:]) - 1  # Extract number from 'p1', 'p2', etc. and adjust to 0-index
                pin2_index = int(pin2_name[1:]) - 1

                pin1 = gate1.pins[pin1_index]
                pin2 = gate2.pins[pin2_index]

                # Create wire connection between pin1 and pin2
                wires.append((pin1, pin2))

    # Step 2: Parse output.txt to update gate locations
    # with open(output_file_path, 'r') as f_out:
    #     for line in f_out:
    #         parts = line.strip().split()
    #         if parts[0].startswith("g"):  # Gate locations
    #             name = parts[0]
    #             x = int(parts[1])
    #             y = int(parts[2])
                
    #             # Find the component with the matching name and update its position
    #             component = next(c for c in components_list if c.name == name)
    #             component.update(x - component.x, y - component.y)

    return components_list, wires, wire_delay, components_dict

class CircuitOptimizer:
    def __init__(self, components_dict, wires, start_temp, cooling_factor, components_list, wire_delay=0):
        self.components_dict = components_dict
        self.components_list = components_list
        self.wires = wires
        self.temperature = start_temp
        self.cooling_factor = cooling_factor
        self.wire_delay = wire_delay


    def calculate_overlap(self):
        overlap = 0
        components_dict_list = list(self.components_dict.values())
        for i in range(len(components_dict_list)):
            for j in range(i + 1, len(components_dict_list)):
                if components_dict_list[i].is_colliding(components_dict_list[j]):
                    overlap += 1
        return overlap

    def attempt_component_relocation(self):
        component = random.choice(list(self.components_dict.values()))
        max_dim = max(max(c.width, c.height) for c in self.components_dict.values())
        grid_size = max_dim * (len(self.components_dict) + 1)  # Ensure enough space
        new_x = random.randint(0, grid_size - component.width)
        new_y = random.randint(0, grid_size - component.height)
        old_x, old_y = component.x, component.y
        component.reposition(new_x, new_y)
        return component, old_x, old_y
    
    def attempt_component_relocation_small(self):
        component = random.choice(list(self.components_dict.values()))
        max_width = max(comp.width for comp in self.components_dict.values())
        max_height = max(comp.height for comp in self.components_dict.values())
        pert = 3*max(max_height, max_width)//2
        old_x, old_y = component.x, component.y
        new_x = old_x + random.randint(-pert,pert)
        new_y = old_y + random.randint(-pert,pert)
        component.reposition(new_x, new_y)
        return component, old_x, old_y
    
    def should_accept_change(self, old_cost, new_cost):
        if new_cost < old_cost:
            return True
        return random.random() < math.exp((old_cost - new_cost) / self.temperature)
        # return False

    def optimize(self):
        print("Starting Annealing")
        path, current_wire_length = self.find_critical_path()
        current_cost = current_wire_length
        best_cost = current_cost
        best_layout = {c.name: (c.x, c.y) for c in self.components_dict.values()}
        print(best_layout)
        if len(best_layout) > 100:
            function = self.attempt_component_relocation_small
        else:
            function = self.attempt_component_relocation

        x = len(self.components_dict.values())
        if x < 20:
            max_iters = 1000000
        elif x < 40:
            max_iters = 500000
        elif x < 100:
            max_iters = 100000
        elif x < 300:
            max_iters = 50000
        else:
            max_iters = 1000
        for _ in tqdm(range(max_iters)):
            component, old_x, old_y = function()
            
            path, new_wire_length = self.find_critical_path()
            new_overlap = self.calculate_overlap()
            new_cost = new_wire_length
            
            if self.should_accept_change(current_cost, new_cost) and not new_overlap:
                # print(new_cost)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_layout = {c.name: (c.x, c.y) for c in self.components_dict.values()}
            else:
                component.reposition(old_x, old_y)
                
            self.temperature *= self.cooling_factor
        

        return best_layout, best_cost, path

    def initial_compact_placement(self, components_dict):
        # print(self.calculate_wire_length())
        n = len(components_dict)
        grid_size = math.ceil(math.sqrt(n))
        
        max_width = max(comp.width for comp in components_dict.values())
        max_height = max(comp.height for comp in components_dict.values())
        
        cell_width = 3*max_width
        cell_height = 3*max_height 
        component_list = list(components_dict.values())
        for i, component in enumerate(component_list):
            row = i // grid_size
            col = i % grid_size
            
            base_x = col * cell_width
            base_y = row * cell_height

            x = base_x
            y = base_y
                
            component.reposition(x,y)

        best_wire_length = self.find_critical_path()[1]
        print("Initial:", best_wire_length)
        best_positions = {id: (comp.x, comp.y) for id, comp in components_dict.items()}
        if len(component_list) < 2:
            return None
        for _ in tqdm(range(50000)):
            i, j = random.sample(range(len(component_list)), 2)
            comp1, comp2 = component_list[i], component_list[j]

            x1, y1 = comp1.x, comp1.y
            x2, y2 = comp2.x, comp2.y

            comp1.reposition(x2, y2)
            comp2.reposition(x1, y1)

            new_wire_length = self.find_critical_path()[1]

            if new_wire_length < best_wire_length:
                print("New:", best_wire_length)
                best_wire_length = new_wire_length
                best_positions = {id: (comp.x, comp.y) for id, comp in components_dict.items()}
            else:
                comp1.reposition(x1, y1)
                comp2.reposition(x2, y2)
        
        for component_id, (x, y) in best_positions.items():
            components_dict[component_id].reposition(x, y)
        
        return None

    
    def find_primary_inputs_outputs(self):
        connected_as_input = set()
        connected_as_output = set()
        
        # Track which pins are connected as inputs and outputs based on wires
        for wire in self.wires:
            source_pin, destination_pin = wire
            connected_as_output.add(source_pin)
            connected_as_input.add(destination_pin)

        
        primary_inputs_gate = []
        
        primary_outputs_gates = []

        # Identify primary inputs and outputs for each component
        for component in self.components_list:
            for pin in component.get_pins():
                # Primary input: Pins on left boundary not connected as input anywhere
                if pin.x == component.x: 
                    if pin in connected_as_input:
                        break
            primary_inputs_gate.append(component)
            
        for component in self.components_list:
            for pin in component.get_pins():
                # Primary output: Pins on right boundary not connected as output anywhere
                if pin.x == component.x + component.width:
                    if pin in connected_as_output:
                        break
            primary_outputs_gates.append(component)
                    
        

        return set(primary_inputs_gate), set(primary_outputs_gates)

    def make_pin_clusters(self):
        output = {}
        
        for wire in self.wires:
            pin1, pin2 = wire
            if pin1 not in output:
                output[pin1] = []
            output[pin1].append(pin2)
            
            
        for i in output:
            l = output[i]
            l.append(i)
            p = PinCluster(l)
            for j in l[:-1]:
                j.prev = i
                j.cluster = p
            i.nxt = l[:-1]
            i.cluster = p
        return output
            
        
    
    def find_critical_path(self):
        # Step 1: Find primary input and output gates
        primary_inputs, primary_outputs = self.find_primary_inputs_outputs()

        # Step 2: Initialize dictionaries to store delays and critical paths
        gate_delays = {}      # Stores the delay for each gate
        pin_delays = {}       # Stores the delay for each pin
        critical_paths = {}   # Stores the critical path leading to each pin/gate

        # Step 3: Helper function to calculate delay for a pin recursively
        def calculate_pin_delay(pin):
            if pin in pin_delays:
                return pin_delays[pin], critical_paths[pin]
            
            if pin.prev is None:
                pin_delays[pin] = 0
                critical_paths[pin] = [pin]
                return 0, [pin]
            
            # Recursively calculate the delay from the previous gate
            prev_gate_delay, prev_gate_path = calculate_gate_delay(pin.prev.gate)
            wire_delay = self.wire_delay * pin.cluster.calculate_wire_length()
            total_delay = prev_gate_delay + wire_delay
            
            pin_delays[pin] = total_delay
            critical_paths[pin] = prev_gate_path + [pin.prev, pin]  # Include both output and input pins
            return total_delay, critical_paths[pin]

        # Step 4: Helper function to calculate delay for a gate recursively
        def calculate_gate_delay(gate):
            if gate in gate_delays:
                return gate_delays[gate], critical_paths[gate]
            
            # Calculate delays for all input pins
            input_pins = [pin for pin in gate.pins if is_input_pin(pin, gate)]
            input_delays_and_paths = [calculate_pin_delay(pin) for pin in input_pins]
            
            # Find the input pin with maximum delay
            max_delay = -float('inf')
            max_delay_path = None
            
            for delay, path in input_delays_and_paths:
                if delay > max_delay:
                    max_delay = delay
                    max_delay_path = path
            
            total_gate_delay = max_delay + gate.delay
            gate_delays[gate] = total_gate_delay
            critical_paths[gate] = max_delay_path  # Store the path leading to this gate
            
            return total_gate_delay, max_delay_path

        # Step 5: Find the primary output with maximum delay
        max_delay = -float('inf')
        critical_path = None
        critical_output = None

        for gate in primary_outputs:
            input_pins = [pin for pin in gate.pins if is_input_pin(pin, gate)]
            for input_pin in input_pins:
                delay, path = calculate_pin_delay(input_pin)
                total_delay = delay + gate.delay
                
                if total_delay > max_delay:
                    max_delay = total_delay
                    critical_path = path
                    critical_output = next(pin for pin in gate.pins if is_output_pin(pin, gate))

        # Add the final output pin to the critical path
        if critical_output:
            critical_path.append(critical_output)

        return critical_path, max_delay
    
    def has_cyclic_connections(self):
        # Step 1: Create helper function for DFS traversal
        def dfs(gate, visited, recursion_stack):
            # Mark the current gate as visited and add to the recursion stack
            visited.add(gate)
            recursion_stack.add(gate)

            # Explore each output pin's next gate connections
            for pin in gate.pins:
                for next_pin in pin.nxt:
                    next_gate = next_pin.gate

                    if next_gate not in visited:  # If the gate hasn't been visited, recurse
                        if dfs(next_gate, visited, recursion_stack):
                            return True
                    elif next_gate in recursion_stack:  # If the gate is already in the recursion stack, cycle detected
                        return True

            # Remove the gate from the recursion stack before backtracking
            recursion_stack.remove(gate)
            return False

        # Step 2: Initialize sets for visited gates and recursion stack
        visited = set()
        recursion_stack = set()

        # Step 3: Run DFS on every gate in the circuit
        for gate in self.components_list:
            if gate not in visited:
                if dfs(gate, visited, recursion_stack):
                    return True  # Cycle detected

        # Step 4: If no cycles are detected
        return False

                

    def calculate_total_wire_length(self):
        total_length = sum(cluster.calculate_wire_length() for cluster in self.pin_clusters)
        return total_length

    def visualize_circuit(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Draw components_list (gates) and annotate their names
        for component in self.components_list:
            # Draw the gate (rectangle)
            rect = patches.Rectangle((component.x, component.y), component.width, component.height,linewidth=1, edgecolor='blue', facecolor='lightblue')
            ax.add_patch(rect)
            
            # Annotate the gate with its name at its bottom-left corner
            ax.text(component.x + component.width / 2, component.y + component.height / 2,
                    component.name, fontsize=12, color='black', ha='center', va='center')

            # Plot the pins
            for pin in component.get_pins():
                ax.plot(pin.x, pin.y, 'ro')  # Plot pin as red dots
                ax.text(pin.x, pin.y, pin.name, fontsize=10, color='black', ha='right')  # Pin name

        # Draw wires between pins and annotate with wire labels (optional)
        for wire in self.wires:
            pin1, pin2 = wire
            ax.plot([pin1.x, pin2.y], [pin1.x, pin2.y], 'g-', linewidth=2)  # Green lines for wires

            # Optional: Annotate wire midway between the two pins
            mid_x = (pin1.x + pin2.x) / 2
            mid_y = (pin1.y + pin2.y) / 2
            ax.text(mid_x, mid_y, 'Wire', fontsize=10, color='green', ha='center')

        # Label axes and add title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title("Circuit Visualization")
        plt.show()


def find_clusters(components_dict, wires):
    # Create an adjacency list of the components_dict based on the wires
    adj_list = defaultdict(set)
    for g1, _, g2, _ in wires:
        adj_list[g1].add(g2)
        adj_list[g2].add(g1)

    visited = set()
    clusters = []

    def dfs(component, cluster):
        visited.add(component)
        cluster.add(component)
        for neighbor in adj_list[component]:
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for component in components_dict.keys():
        if component not in visited:
            current_cluster = set()
            dfs(component, current_cluster)
            clusters.append(current_cluster)
    # print(clusters)
    return clusters

def calculate_bounding_box(components):
    min_x = min(c.x for c in components.values())
    min_y = min(c.y for c in components.values())
    max_x = max(c.x + c.width for c in components.values())
    max_y = max(c.y + c.height for c in components.values())
    return (max_x, max_y)

def output_file(file_path, best_layout,components_dict, path, delay):
    bb = calculate_bounding_box(components_dict)
    with open(file_path, 'w') as out:
        out.write(f"bounding_box {bb[0]} {bb[1]}\n")
        out.write("critical_path ")
        for i in path:
            k = i.name.split(".")
            out.write(f"{k[0]}.p{k[1]} ")
        out.write("\n")
        out.write(f"critical_path_delay {delay}\n")

        for component in components_dict.values():
            out.write(f"{component.name} {component.x} {component.y}\n")

    return None

        # bounding box 
        # critical path
        # critical delay
    


# Example Usage
input_file_path = "D:/col215/sw3/2023CS10807_2023CS10804/input1.txt"  # Path to input file
output_file_path = "D:/col215/sw3/2023CS10807_2023CS10804/answers/output1.txt"  # Path to output file
components_list, wires, wire_delay, components_dict = parse_input_file(input_file_path, output_file_path)

start_temp, cooling_factor = 1000, 0.95


x_offset = 0


optimizer = CircuitOptimizer(components_dict, wires, start_temp, cooling_factor, components_list, wire_delay)

optimizer.make_pin_clusters()
if optimizer.has_cyclic_connections(): 
    print("Has Cyclic Connections")
else:
    optimizer.initial_compact_placement(components_dict)
    best_layout, delay, path = optimizer.optimize()

    x_min = min(x for name, (x, _) in best_layout.items())
    cluster_width = max(components_dict[name].width + x - x_min for name, (x, _) in best_layout.items())
    y_min = min(y for name, (_, y) in best_layout.items())
    for name, (x, y) in best_layout.items():
        components_dict[name].reposition(x + x_offset - x_min, y - y_min)

    x_offset += cluster_width  

    output_file(output_file_path, best_layout,components_dict, path, delay)
    print(f"Final delay: {delay}")



    
