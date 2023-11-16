import random

# Initialize a sorted sequence
sorted_sequence = sorted([random.randint(1, 100) for _ in range(10)])
print(f"Initial sorted sequence: {sorted_sequence}")

# Insert a sequence of random integers
for _ in range(5):
    insert_pos = random.randint(0, len(sorted_sequence))
    print(insert_pos)
    sorted_sequence.insert(insert_pos, random.randint(1, 100))
sorted_sequence.sort()
print(f"After inserting random integers: {sorted_sequence}")

# Remove elements as determined by a random sequence of positions
positions = sorted(random.sample(range(len(sorted_sequence)), 5), reverse=True)
print(positions)
for pos in positions:
    del sorted_sequence[pos]
print(f"After removing elements at random positions: {sorted_sequence}")