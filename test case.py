import random
from collections import defaultdict

def generate_test_case_with_patterns(pattern, num_gates=50, total_wires=200, wire_delay=4):
    gates = []
    wires = []
    pin_counts = {}
    input_pins = defaultdict(list)
    output_pins = defaultdict(list)

    gates.append(f"wire_delay {wire_delay}")
    
    # Step 1: Initialize Gates and Assign Pins
    for i in range(1, num_gates + 1):
        width = random.randint(2, 25)
        height = random.randint(10, 25)
        delay = random.randint(1, 10)  # Random delay for each gate
        num_pins = min(height - 1, 20)  # Ensure number of pins doesn't exceed height
        gates.append(f"g{i} {width} {height} {delay}")
        pin_counts[f'g{i}'] = num_pins

        pins = []
        input_pin_list = []
        output_pin_list = []
        for j in range(1, num_pins + 1):
            # Assign pins to left edge (input) or right edge (output)
            if j % 2 == 0:
                pin_x = 0  # Left edge (input pin)
                input_pin_list.append(f"g{i}.p{j}")
            else:
                pin_x = width  # Right edge (output pin)
                output_pin_list.append(f"g{i}.p{j}")
            pin_y = random.randint(0, height - 1)
            pins.append(f" {pin_x} {pin_y}")
            
        output_pins[f'g{i}'] = output_pin_list
        input_pins[f'g{i}'] = input_pin_list
        
        pin_line = " ".join(pins)
        gates.append(f"pins g{i} {pin_line}")

    # Step 2: Create Different Types of Graphs Based on the Pattern
    connections = []

    if pattern == "random":
        while len(connections) < total_wires:
            start_gate = f'g{random.randint(1, num_gates)}'
            end_gate = f'g{random.randint(1, num_gates)}'
            if start_gate != end_gate and output_pins[start_gate] and input_pins[end_gate]:
                output_pin = random.choice(output_pins[start_gate])
                input_pin = random.choice(input_pins[end_gate])
                connections.append((start_gate, end_gate, output_pin, input_pin))

    elif pattern == "sparse":
        for _ in range(total_wires):
            start_gate = f'g{random.randint(1, num_gates)}'
            end_gate = f'g{random.randint(1, num_gates)}'
            if start_gate != end_gate and output_pins[start_gate] and input_pins[end_gate]:
                if random.random() < 0.05:  # Sparse probability
                    output_pin = random.choice(output_pins[start_gate])
                    input_pin = random.choice(input_pins[end_gate])
                    connections.append((start_gate, end_gate, output_pin, input_pin))

    elif pattern == "dense":
        for i in range(1, num_gates + 1):
            for j in range(i + 1, num_gates + 1):
                if len(connections) >= total_wires:
                    break
                start_gate = f'g{i}'
                end_gate = f'g{j}'
                if output_pins[start_gate] and input_pins[end_gate]:
                    output_pin = random.choice(output_pins[start_gate])
                    input_pin = random.choice(input_pins[end_gate])
                    connections.append((start_gate, end_gate, output_pin, input_pin))

    elif pattern == "chain":
        for i in range(1, num_gates):
            start_gate = f'g{i}'
            end_gate = f'g{i + 1}'
            if output_pins[start_gate] and input_pins[end_gate]:
                output_pin = random.choice(output_pins[start_gate])
                input_pin = random.choice(input_pins[end_gate])
                connections.append((start_gate, end_gate, output_pin, input_pin))

    elif pattern == "star":
        center = f'g{random.randint(1, num_gates)}'
        for i in range(1, num_gates + 1):
            if f'g{i}' != center:
                start_gate = center
                end_gate = f'g{i}'
                if output_pins[start_gate] and input_pins[end_gate]:
                    output_pin = random.choice(output_pins[start_gate])
                    input_pin = random.choice(input_pins[end_gate])
                    connections.append((start_gate, end_gate, output_pin, input_pin))

    index_map = {f'g{i + 1}': i for i in range(num_gates)}
    reachability = [[float('inf')] * num_gates for _ in range(num_gates)]

    for i in range(num_gates):
        reachability[i][i] = 0

    for start_gate, end_gate, _, _ in connections:
        u = index_map[start_gate]
        v = index_map[end_gate]
        reachability[u][v] = 1

    for k in range(num_gates):
        for i in range(num_gates):
            for j in range(num_gates):
                if reachability[i][j] > reachability[i][k] + reachability[k][j]:
                    reachability[i][j] = reachability[i][k] + reachability[k][j]

    final_wires = []
    input_pin_connections = set()
    for start_gate, end_gate, output_pin, input_pin in connections:
        u = index_map[start_gate]
        v = index_map[end_gate]
        if reachability[u][v] == 1:
            final_wires.append(f"wire {output_pin} {input_pin}")
            input_pin_connections.add(input_pin)

    return gates, final_wires

def write_to_file(filename, gates, wires):
    with open(filename, "w") as file:
        file.write("\n".join(gates))
        file.write("\n")
        file.write("\n".join(wires))

# Generate test case using the new approach
patterns = ["random", "sparse", "dense", "chain", "star"]

for pattern in patterns:
    gates, wires = generate_test_case_with_patterns(pattern)
    write_to_file(f"{pattern}_input.txt", gates, wires)
    print(f"Generated test case for {pattern} pattern")
