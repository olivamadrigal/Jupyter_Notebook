# QUISKIT CODE CELLS WITH QUICK EXERCISES. PASTE 1ST BLOCK AND SUBSEQUENT BLOCKS
# SEPARETELY INTO JUPYTER NOTEBOOK.
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector, plot_state_qsphere
import numpy as np
from math import sqrt, pi
import cmath
from qiskit_textbook.widgets import state_vector_exercise, gate_demo
from qiskit_textbook.widgets import bloch_calc
from qiskit_textbook.widgets import plot_bloch_vector_spherical
from qiskit_textbook.tools import array_to_latex
from qiskit.circuit import Gate

#n_q = 8  # number of qubits
#n_b = 8  # number of output bits extracted
#qc_output = QuantumCircuit(n_q, n_b)
##we extract using measure
#for j in range(8):  # 0 to 7
#    qc_output.measure(j, j)  # qubit j write output to bit j
#print(qc_output.draw(output='text'))
#
#sim = Aer.get_backend('qasm_simulator')  # this is the simulator we'll use
#qobj = assemble(qc_output)  # this turns the circuit into an object our backend can run
#result = sim.run(qobj).result()  # we run the experiment and get the result from that experiment
## from the results, we get a dictionary containing the number of times (counts)
## each result appeared
#counts = result.get_counts()
## and display it on a histogram
#plot_histogram(counts)
#
#n = 8
#qc_encode = QuantumCircuit(n)
#qc_encode.x(7)  # x == NOT gate
#qc_encode.draw()
#
#qc = qc_encode + qc_output  # extract results
#qc.draw()
#
#qobj = assemble(qc)
#counts = sim.run(qobj).result().get_counts()
#plot_histogram(counts)
#
#qc_encode = QuantumCircuit(n)
#qc_encode.x(1)
#qc_encode.x(5)
#qc_encode.draw()
#
#
#qc_cnot = QuantumCircuit(2)
#qc_cnot.cx(0,1)
#qc_cnot.draw()
#
#qc = QuantumCircuit(2,2)
#qc.x(0)
#qc.cx(0,1)
#qc.measure(0,0)
#qc.measure(1,1)
#qc.draw()
#
#qc_ha = QuantumCircuit(4,2)
## encode inputs in qubits 0 and 1
#qc_ha.x(0) # For a=0, remove this line. For a=1, leave it.
#qc_ha.x(1) # For b=0, remove this line. For b=1, leave it.
#qc_ha.barrier()
## use cnots to write the XOR of the inputs on qubit 2
#qc_ha.cx(0,2)
#qc_ha.cx(1,2)
#qc_ha.barrier()
## extract outputs
#qc_ha.measure(2,0) # extract XOR value
#qc_ha.measure(3,1)
#qc_ha.draw()
#
#qc_ha = QuantumCircuit(4,2)
## encode inputs in qubits 0 and 1
#qc_ha.x(0) # For a=0, remove the this line. For a=1, leave it.
#qc_ha.x(1) # For b=0, remove the this line. For b=1, leave it.
#qc_ha.barrier()
## use cnots to write the XOR of the inputs on qubit 2
#qc_ha.cx(0,2)
#qc_ha.cx(1,2)
## use ccx to write the AND of the inputs on qubit 3
#qc_ha.ccx(0,1,3)
#qc_ha.barrier()
## extract outputs
#qc_ha.measure(2,0) # extract XOR value
#qc_ha.measure(3,1) # extract AND value
#qc_ha.draw()
#
#qobj = assemble(qc_ha)
#counts = sim.run(qobj).result().get_counts()
#plot_histogram(counts)  # .show() etc not needed in notebook, it's automatic :)
#
##  REPRESENTING QUBIT STATES
#qc = QuantumCircuit(1) # Create a quantum circuit with one qubit
##  In our quantum circuits, our qubits always start out in the state |0⟩
##  We can use the initialize() method to transform this into any state.
#initial_state = [0,1]   # Define initial_state as |1>
#qc.initialize(initial_state, 0) # Apply initialisation operation to the 0th qubit
#qc.draw()  # Let's view our circuit
#
#svsim = Aer.get_backend('statevector_simulator') # Tell Qiskit how to simulate our circuit
#qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit; redefine...
#initial_state = [0,1]   # Define initial_state as |1>
#qc.initialize(initial_state, 0) # Apply initialisation operation to the 0th qubit
#qobj = assemble(qc)     # Create a Qobj from the circuit for the simulator to run
#result = svsim.run(qobj).result() # Do the simulation and return the result
#out_state = result.get_statevector()
#print(out_state) # Display the output state vector
#
#qc.measure_all()
#qc.draw()
#qobj = assemble(qc)
#result = svsim.run(qobj).result()
#counts = result.get_counts()
#plot_histogram(counts) # P(|1>) = 100%
#
## put q0 into supersposition and probe 1000 times
#initial_state = [1/sqrt(2), 1j/sqrt(2)]  # Define state |q_0>
#qc = QuantumCircuit(1) # Must redefine qc
#qc.initialize(initial_state, 0) # Initialise the 0th qubit in the state `initial_state`: supers.
#qobj = assemble(qc)
#state = svsim.run(qobj).result().get_statevector() # Execute the circuit
#print(state)           # Print the result
#
#results = svsim.run(qobj).result().get_counts()
#plot_histogram(results) # P(supers) = 50:50 for |0> and |1>
##  We can see we have equal probability of measuring either |0> or |1>
##  To explain this, we need to talk about measurement.
#state_vector_exercise(target=1/3)
#
#qc = QuantumCircuit(1) # We are redefining qc
#initial_state = [0.+1.j/sqrt(2),1/sqrt(2)+0.j]
#qc.initialize(initial_state, 0)
#qc.draw()
#
#qobj = assemble(qc)
#state = svsim.run(qobj).result().get_statevector()
#print("Qubit State = " + str(state))
#
## the act of measure changes the state
#qc.measure_all()
#qc.draw()
#qobj = assemble(qc)
#state = svsim.run(qobj).result().get_statevector()
#print("State of Measured Qubit = " + str(state))
#
## block sphere 3d representation of the qubit
#coords = [pi/2,0,1] # [Theta, Phi, Radius]
#plot_bloch_vector_spherical(coords).show() # Bloch Vector with spherical coordinates
#bloch_calc()

####### GATES ######
# Let's do an X-gate on a |0> qubit
qc = QuantumCircuit(1)
qc.x(0)
qc.draw()
# Let's see the result
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
state = svsim.run(qobj).result().get_statevector()
plot_bloch_multivector(state)

# Run the code in this cell to see the widget
from qiskit_textbook.widgets import gate_demo
gate_demo(gates='pauli')

qc.y(0) # Do Y-gate on qubit 0
qc.z(0) # Do Z-gate on qubit 0
qc.draw()

# Run the code in this cell to see the widget
gate_demo(gates='pauli+h')

# play with initialize in an X-basis eigen vector.
# if initial state is |->, then H\-> = |1> and if initial is |+>, then H|+> = |0>.
initial_state = [1/sqrt(2), -(1/sqrt(2))]  # INITIALIZE TO |-> eigenvector from X-basis. [ 0.70710678+0.j -0.70710678+0.j]
#initial_state = [1/sqrt(2), 1/sqrt(2)]  # INITIALIZE TO |+> eigenvector from X-basis.
qc = QuantumCircuit(1,1) # single qubit
qc.initialize(initial_state, 0)  #initialize it.
qc.h(0)  # Applying Hadamard gate brings it to |0> or |1>
qc.measure(0, 0) # Then we measure
# run simulation
qasmsim = Aer.get_backend('qasm_simulator')  # Tell Qiskit how to simulate our circuit
svsim = Aer.get_backend('statevector_simulator') # Tell Qiskit how to simulate our circuit
qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
counts = qasmsim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
sv1 = svsim.run(qobj).result().get_statevector()
print(sv1)  # we see the supersposition is destroyed... after we measure since [0 val] or [val 0]
plot_histogram(counts)  # Display the output state vector
#[1.+0.j 0.+0.j] = state |0>
#[0.+0.j 1.+0.j] = state |1>
#[0.+0.000000e+00j 1.-6.123234e-17j] => [0+0j 1 + -6.12e^-17j]   # supersposition
#[ 1.-6.123234e-17j -0.+0.000000e+00j]
qc.h(0)  # Apply H again
qc.measure(0, 0) # measure again
qobj = assemble(qc) # reassemble
counts2 = qasmsim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
sv2 = svsim.run(qobj).result().get_statevector()
print(sv2)
plot_histogram(counts2)  # Display the output state vector
# in combinataionl circuit we would expect for initial input |->, we get first |1> as output,
# then with feedback loop input of |1>, we would get a |-> output from our H-gate circuit.
# in this case, with qubits, the qubit is brough to supersposition, measurement destroys that,
# we see
# the measured output is either 1 or 0,


# displays probabilites from intial state of |->, or |+>
qc1 = QuantumCircuit(1,1) # single qubit
qc1.initialize([1/sqrt(2), (1/sqrt(2))], 0)  # |+>
qc1.h(0) # H gate
qc1.measure(0, 0) # Then we measure
qc2 = QuantumCircuit(1,1) # single qubit
qc2.initialize([1/sqrt(2), -(1/sqrt(2))], 0)  # |->
qc2.h(0) # H gate
qc2.measure(0, 0) # Then we measure
# run simulation
qasmsim = Aer.get_backend('qasm_simulator')
svsim = Aer.get_backend('statevector_simulator')
qc1_obj = assemble(qc1)
qc2_obj = assemble(qc2)
cnt1 = qasmsim.run(qc1_obj).result().get_counts()
cnt2 = qasmsim.run(qc2_obj).result().get_counts()
sv_o1 = svsim.run(qc1_obj).result().get_statevector()
sv_o2 = svsim.run(qc2_obj).result().get_statevector()
plot_histogram(cnt1)
plot_histogram(cnt2)
print(sv_o1)
print(sv_o2)

# displays probabilites from intial state of |->, or |+>
qasmsim = Aer.get_backend('qasm_simulator')
svsim = Aer.get_backend('statevector_simulator')
qc = QuantumCircuit(1,1) # single qubit
qc.initialize([1/sqrt(2), (1/sqrt(2))], 0)  # init to |+>
qc.h(0) # H gate
qc.measure(0, 0)
# run simulation
qcobj = assemble(qc)
counts = qasmsim.run(qcobj).result().get_counts()
sv = svsim.run(qcobj).result().get_statevector()
print(sv)
plot_histogram(counts, color='pink')

# displays probabilites from intial state of |->, or |+>
qasmsim = Aer.get_backend('qasm_simulator')
svsim = Aer.get_backend('statevector_simulator')
qc = QuantumCircuit(1,1) # single qubit
qc.initialize([1/sqrt(2), -1/sqrt(2)], 0)  # init to |+>
qc.h(0) # H gate
qc.measure(0, 0)
# run simulation
qcobj = assemble(qc)
counts = qasmsim.run(qcobj).result().get_counts()
sv = svsim.run(qcobj).result().get_statevector()
print(sv)
plot_histogram(counts, color='gray')

# MEASUREMENT from state in the Y-basis
# EIGEN VECTORS IN Y-BASIS: v1= 1/√2[1, i], v2=1/√2[-i, 1]
qasmsim = Aer.get_backend('qasm_simulator')
svsim = Aer.get_backend('statevector_simulator')
qc = QuantumCircuit(1,1) # single qubit
#v1= 1/√2[1, i] = 1/√2(|1> + i|0>)
#init_vector = [1/sqrt(2)+0.j, 0.+1.j/sqrt(2)]
#v2=1/√2[-i, 1]
init_vector = [0.-1.j/sqrt(2), 1/sqrt(2)+0.j]
qc.initialize(init_vector, 0)  # initialize to eigenvector in Y-basis
qc.h(0) # apply H-gate
qc.measure(0, 0) # Then we measure
# run simulation
qcobj = assemble(qc)
counts = qasmsim.run(qcobj).result().get_counts()
sv = svsim.run(qcobj).result().get_statevector()
print(sv)
plot_histogram(counts, color='yellow')

#More generally: Whatever state our quantum system is in, there is always a measurement that has a deterministic outcome.


### R-GATE
# Run the code in this cell to see the widget
gate_demo(gates='pauli+h+rz')
qc = QuantumCircuit(1)
qc.rz(pi/4, 0)
qc.draw()

#### SPECIAL CASES OF R-GATE: I, S, & T
# I=IDENTITY MATRIX, whose eigenvectors are the X-basis. I = XX.
# Z = R for phi=pi.
# S = √Z-gate for phi=pi/2; does a quarter-turn around the Bloch sphere;
# NOTE: SS|q> = Z|q>
# S-dagger-gate = R for phi= - pi/2.

# S-GATE
qc = QuantumCircuit(1)
qc.s(0)   # Apply S-gate to qubit 0
qc.sdg(0) # Apply Sdg-gate to qubit 0
qc.draw()

# T-gate is R with phi = pi/4; dagger with phi = -pi/4
qc = QuantumCircuit(1)
qc.t(0)   # Apply T-gate to qubit 0
qc.tdg(0) # Apply Tdg-gate to qubit 0
qc.draw()


# U3(theta, phi, lamdba) general gate:  X,Z,Y, and R-phi  gates all can be expressed
# in terms of this. Special cases U1, U2. All SINGLE QUBIT GATES get compiled down to
# U1, U2, or U3 before executing on real hardware.

#With qubits and quantum gates, we can design novel algorithms that are fundamentally different from digital and analog classical ones. In this way, we hope to find solutions to problems that are intractable for classical computers.
#With a quantum computer, however, the fact that we can create superposition states means that the function can be applied to many possible inputs simultaneously. This does not mean that we can access all possible outputs since measurement of such a state simply gives us a single result. However, we can instead seek to induce a quantum interference effect, which will reveal the global property we require.

#quantum search algorithm will always out-perform the classical search algorithm.: grover
#Grover: from O(n) to O(√n) => quadratic speed up.

#An even more impressive speedup is obtained with Shor's algorithm, which analyses periodic functions at the heart of the factorization problem.
#Shor: from exponential to superpolynomial (cubic): O(e^(n^1/3)) => O(n^3).

#quantum algorithms is to use quantum computers to solve quantum problems.

#Particularly promising are those problems for which classical algorithms face inherent scaling limits and which do not require a large classical dataset to be loaded.

#For quantum advantage, a given problem's answers need to strongly depend on exponentially many entangled degrees of freedom with structure such that quantum mechanics evolves to a solution without having to go through all paths. Note, however, that the precise relationship between problems that are 'easy' for quantum computers (solvable in polynomial time) and other complexity-theoretic classes is still an open question


# true power of quantum computing is realised through the interactions between qubits.

# In classical digital electronics: AND, OR, NOT: gates to build any classical algorithm
# In quantum regime: basic quantum gates: buildingblocks to build any quantum algorithm.


# https://people.eecs.berkeley.edu/~vazirani/s09quantum/notes/lecture2.pdf
#An extreme case of this phenomenon occurs when we consider an n qubit quantum system. The Hilbert n
#space associated with this system is the n-fold tensor product of Cˆ2 ≡ Cˆ2 . Thus nature must “remember” of 2n complex numbers to keep track of the state of an n qubit system. For modest values of n of a few hundred, 2n is larger than estimates on the number of elementary particles in the Universe.


################################################################################################
# IN GENERAL: n-bits can be used to encode one of 2^n different values in [0,(2^n) - 1]
# at a given point in time.

# n qubits can be used to encode all 2^n different classical values in [0,(2^n) - 1].
# due to supersposition, in a quantum system, there is a conitum of possible values qubits can take
# since they can be in supersposition.

# however, a quantum system does not need to explore all possible values to find a solution.
# to accomplish this, depends on how we work with the qubits, usually adhering to a property
# or properties specified by the algorithm. this is were we obtain the real power from quantum machines.
################################################################################################

#%%%%%%%%%%%%%%%%%%%%%%% MULTIQUBIT SYSTEMS %%%%%%%%%%%%%%%%%%%%%%%%%%
qc = QuantumCircuit(3)
# Apply H-gate to each qubit:
for qubit in range(3):
    qc.h(qubit)
# See the circuit:
qc.draw()
# Let's see the result
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()

# In Jupyter Notebooks we can display this nicely using Latex.
# If not using Jupyter Notebooks you may need to remove the
# array_to_latex function and use print(final_state) instead.
#array_to_latex(final_state, pretext="\\text{Statevector} = ")
print(final_state)

#Single Qubit Gates on Multi-Qubit Statevectors
qc = QuantumCircuit(2)  # 2 qubits circuit...
qc.h(0)  # single gate hadamard gate matrix apply to qubit 0
qc.x(1)  # single gate NOT gate matrix apply to qubit 1
qc.draw()

# we can express the simulataneous operation of both gates on
# on the circuit via thier tensor product.
#
#the unitary simulator multiplies all the gates in our circuit together to compile a single unitary matrix that performs the whole quantum circuit:
# X(tensorproduct)H:
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()


# apply a gate to only one qubit at a time, use tensor product with the identity matrix, e.g.:
# X < tensor product> I.
qc = QuantumCircuit(2)
qc.x(1)
qc.draw()
# Simulate the unitary
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
# Display the results:
array_to_latex(unitary, pretext="\\text{Circuit = } ")

#exercies:
# Calculate the (single qubi) unitary U created by the sequence of gates U = XZH.
# these are all single qubit gates, they operate on a single qubit.
# U = XZH. matrix multiplication of these matrices.
# Try changing the gates in the circuit above. Calculate their tensor product.
#  H(tensor)Z(tensor)U.
# playing with sample circuit below
qc = QuantumCircuit(2)
qc.x(0)
qc.z(0)
qc.h(0)
qc.draw()
# Simulate the unitary from applying the series of gates on |q0> of multi-qubit system qc consisting
# of two qubits
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, pretext="\\text{Circuit = } ")
# represent the state of multiple qubits with the tensor product
# represent the collective series of gates in a circuit also with their respective tensor prodict.
# difference: when all applied to one qubit, or different qubits..
# x, h, z: qiskit computes the tensor product in reverse order that coded.


# 2-INPUT QUANTUM GATES: [4 by 4] unitaries.
# CNOT: apply NOT to target if control is |1>
# follow simple table when both qubits are in a basis state of |1> or |0>
# control and target qubits:
# if control qubit == |1>: complement the target qubit (apply X(not) gate).
# both control and target qubits are in basis states |0> or |1>
# CNOT is 2-input qubit gate representated by one two 4by4 unitary matrices.
qc = QuantumCircuit(2)
# Apply CNOT
qc.cx(0,1)
# See the circuit:
qc.draw()


# Lets just put one qubit in supersposition and see the collective vector
# state of this two qubit system.
# 2 qubit system: initialize both to |0>
qc = QuantumCircuit(2)
# Apply H-gate to the first and thus put |q0> in state |+>
qc.h(0)
qc.draw()
# Let's see the result: brings our collective state vector
# representing both qubits as |0+>
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
# Print the statevector neatly:
array_to_latex(final_state, pretext="\\text{Statevector = }")

# Now, what happends with this 2-input CNOT gate when we apply it to
# the system and one of the states is in supersposition?
# same experiment but apply CNOT q1.
qc = QuantumCircuit(2)
# Apply H-gate to the first:
qc.h(0)  # put q0 in supersposition |+>
# Apply a CNOT:
qc.cx(0,1)  # q0= control, q1=target
qc.draw()
# Let's get the result:
qobj = assemble(qc)
result = svsim.run(qobj).result()
# Print the statevector neatly:
final_state = result.get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector = }")
# we get an ENTAGLED STATE!!!!!!!!!!!! => shared states cannot be used for communication.
plot_bloch_multivector(final_state)
# entangled state cannot be expressed as two separate state vectors: if we plot on bloch, info is lost.
# if we measure, we know this is random. for some cases, some measurements are deterministic as we have seen.
# however, if we measure only single qubits, then  the correlation is lost.
# moreover, different entangled state vectors cannot be distinguished...
# Q SPEHRE WORKS WELL TO VISUALIZE THE 2-qubit 4-d vector...
plot_state_qsphere(final_state)

# exercise: create circuit that generates entangled state: 1/√2(|01> + |10>) = [0110]1/√2
qc = QuantumCircuit(2) #qubits are set to |0>
qc.x(1) # q1 is now |1>
qc.h(0) # q0 is now 1/√2[1,1]
display(qc.draw()) # collective state of 1/√2[0011]
qc.cnot(0,1) # let q0 be the control.
display(qc.draw()) # collective state of 1/√2[0011]
qobj = assemble(qc)
result = svsim.run(qobj).result()
# Print the statevector neatly:
final_state = result.get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector = }")
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, pretext="\\text{Circuit = } ")
# CNOT = (right matrix in chapter 2 sec 3.1) if control is q1
# the unitary for this circuit U = CNOT•(X⊗H) where • is normal matrix
# multiplication with of rows of CNOT and columns of the tensor product.
# The CNOT matrix we use depends on the control qubit...
# if we test U|00> = we get the desired bell state.


# YES, if control = |+>, then we can put the two qubits in an entangled state.
# now, what if both qubits are in supersposition?
qc = QuantumCircuit(2)
qc.h(0) # |+>
qc.h(1) # |+>, collective |++> = 1/2(|00> + |01> + |10> + |11>)
qc.cx(0,1)  # apply cnot to |++> swaps |01> and |11>, linear sum is the same, no change.
display(qc.draw())  # `display` is a command for Jupyter notebooks
                    # similar to `print`, but for rich content
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)


# we but q1 in |-> so it has a - phase.
qc = QuantumCircuit(2)
qc.h(0) # |+>
qc.x(1) # |+>, collective |++> = 1/2(|00> + |01> + |10> + |11>)
qc.h(1)
display(qc.draw())  # `display` is a command for Jupyter notebooks
                    # similar to `print`, but for rich content
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# if we compute CNOT|-+> = we get |--> swapping |01> and |11> amplitudes
# suprise: affects control qubit state but target is not affected.
qc.cx(0,1)
display(qc.draw())

qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)

# because H gate transforms |+> into |0> and |-> into |1>,
# if we wrap CNOT in H gates (feed it output of Hgates and apply H-gates to its outputs)
# this has effect of CNOT acting opposite: with q1 as control and q0 as target.
# VERIFY
qc = QuantumCircuit(2)
# DOING THIS:
qc.h(0) # |+>
qc.h(1) # |+>
qc.cx(0,1) # CONTROL IS Q0 CNOT|++> = |++>
qc.h(0) # |0>
qc.h(1) # |0>
display(qc.draw())
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, pretext="\\text{Circuit = }\n")
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
# we see the collective state vector of |00>.
# no oppposite effect reflected since q0 or q1 have no effect as control.
# IS THE SAME AS THE ORIGNAL INPUTS AND FLIPING THE CONTROL QUBIT
# CNOT|00>
qc = QuantumCircuit(2)
qc.cx(1,0) # CONTROL IS Q1, result is the same |00>
display(qc.draw())
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, pretext="\\text{Circuit = }\n")
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)

# CONTROLLED-T
# Rotation by phi gates. controlled t-gate
# def cp(theta, control_qubit, target_qubit)
qc = QuantumCircuit(2)
qc.cp(pi/4, 0, 1)  #control is q0, target is q1
display(qc.draw())
# See Results:
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, pretext="\\text{Controlled-T} = \n")
svsim = Aer.get_backend('statevector_simulator')
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)

# VIEW OF |1+>
qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
plot_bloch_multivector(final_state)

# now: Ctrl-T|1+>
# WE CAN SEE CONTROL Q0 is rotated around-Z by pi/4 but target q1 is unaffected.
# There is no clear control or target qubit for all cases.
qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
# Add Controlled-T
qc.cp(pi/4, 0, 1)
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
plot_bloch_multivector(final_state)

#  QUICK EXERCIES #1 pre
svsim = Aer.get_backend('statevector_simulator')
qc = QuantumCircuit(2)
qc.h(0) #u1(0,0,lamda) = r_phi = r_gate with phi=-pi/2 as stated in the problem.
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
#  QUICK EXERCIES #1 post controlled-u1 gate
svsim = Aer.get_backend('statevector_simulator')
qc = QuantumCircuit(2)
qc.h(0) #u1(0,0,lamda) = r_phi = r_gate with phi=-pi/2 as stated in the problem.
#qc.rz(pi/4, 0)  # APPLY R-z gate with phi = pi/2.
qc.cu1(pi/4,0,1)
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# nothing happends to control. CONTROLLED-U matrix and statevector of |0+>
# cancels the phase term. result is [1 1 0 0]1/√2


#  QUICK QUESTION # 2
# What would happen to the control qubit (q0) if the if the target qubit (q1) was
# in the state |1> and the circuit used a controlled-Sdg gate instead of the controlled-T?
# 2 ------before gate
qc = QuantumCircuit(2)
qc.x(1) # q1 = |1>
qc.h(0) # q0 = |+>
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
#2 ----- after apply gate
qc = QuantumCircuit(2)
qc.x(1) # q1 = |1>
qc.h(0) # q0 = |+>
# Add Controlled-Sdagger
qc.cu1(-pi/2, 0, 1)  # controlled-phase gate with phi=-pi/2 (sdag), controll is q0
display(qc.draw())
# See Results:
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state) # we see -90 rot around z-axis as expected


#  QUICK QUESTION # 3
# pre
qc = QuantumCircuit(2)
qc.x(0) #|1>
qc.x(1) #|1>
display(qc.draw())
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# post
qc = QuantumCircuit(2)
qc.x(0) #|1>
qc.x(1) #|1>
qc.cp(pi/4, 0, 1)  #controlled-T, control is q0
display(qc.draw())
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# global phase is not observable, :. we see q0 unaffected.


#More Circuit Identities
qc = QuantumCircuit(2)
c = 0
t = 1
#CONTROLLED Z-GATE
qc = QuantumCircuit(2)
c = 0
t = 1
#When we program quantum computers, our aim is always to build useful quantum circuits from the basic building blocks
#how we can transform basic gates into each other, and how to use them to build some gates that are slightly more complex (but still pretty basic).
qc.cz(c, t)
display(qc.draw())
#In IBM Q devices, however, the only kind of two-qubit gate that can be directly applied is the CNOT
#Hadamard transforms |0> and |1> to |+> and |->.
#Zgate on |+> and |-> is the same as X on |0> and |1>.
#using matrix multiplication: HXH = Z, and HZH=X... hence
#we can use these tricks to transform CNOT into a Ctrl-Z gate:
# CNOT-->Ctrl-Z: CNOT(c,t), H(t). Transform the X(t) into a Z(t)
qc.h(t)
qc.cx(c,t)
qc.h(t)
# same as having applied Z matrix on q_t but in a controlled manner
# depending on q_c.
display(qc.draw())


# TRANSFORM single CNOT into a controlled version of any rotation around the Bloch sphere
# by an angle π: by simply preceding and following it with the correct rotations
# CONTROLLED-Y:
# TRANSFORM single CNOT into a controlled version of any rotation around the Bloch sphere
# by an angle π: by simply preceding and following it with the correct rotations
qc = QuantumCircuit(2)
# SDG(X)S = Controlled-Y.
qc.sdg(1)  # S-dagger on q_t rot on Z by pi/2
qc.cx(0,1) # X
qc.s(1) # S rot on q_t by -pi/2.
display(qc.draw())

# CONTROLLED-H: play and see effects commenting out and uncommenting...
qc = QuantumCircuit(2)
# a controlled-H
qc.x(0) # if control is |1>
qc.ry(pi/4,t) # positive rotation around Y-axis by pi/4 for target
qc.cx(0,1) # if q0=|1>, x(t)
qc.ry(-pi/4,t) # then rotation around Y-axis by pi/4 for target
qc.draw()
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)

#2. Swapping Qubits
# Sometimes we need to move information around in a quantum computer.
#For some qubit implementations, this could be done by physically moving them.
#Another option is simply to move the state between two qubits. This is done by the SWAP gate.
# SWAP using our standard gate set.# SWAP using our standard gate set.
# case: a=|1> and b=|0>
a = 0
b = 1
qc = QuantumCircuit(2)
# swap a 1 from a to b
qc.cx(a,b) # control is a; => b from |0> to |1>
qc.cx(b,a) # since b =|1> = control: a from |1> to |0>
display(qc.draw())
# given a in |0> and b in |1>,
# SWAP back to the original one.
qc.cx(b,a) # puts a in |1>
qc.cx(a,b) # puts b in |0>
display(qc.draw())
# combine fwd and bckwds: 1st or 3rd gate is useless
# but we can swap in from |0> and |1> or vice versa.
qc = QuantumCircuit(2)
qc.cx(b,a)
qc.cx(a,b)
qc.cx(b,a)
display(qc.draw())
# works for all states in computational basis and therefore all general ones.
# same effect if reverse order of gates
qc = QuantumCircuit(2)
# swaps states of qubits a and b
qc.cx(a,b)
qc.cx(b,a)
qc.cx(a,b)
qc.draw()


# QUCIK EXERCISE: Find different circuit that swaps qubits in the states
# https://www.cl.cam.ac.uk/teaching/1617/QuantComp/slides4.pdf
# |+> and |->
# eigenvectors of Hadamard gate
he1 = [1/sqrt(2), 1/sqrt(2)]
he2 = [1/sqrt(2), -1/sqrt(2)]
qc = QuantumCircuit(2)
qc.initialize(he1,0)  #q0 = |+>
qc.initialize(he2,1)  #q1 = |->
# flip one way
qc.ry(pi,1) # rotate by pi around y-axis: q1: |+> in |->
qc.ry(-pi,0) # rotate by -pi around y-axis: q0: |+> in |->
# flip back
qc.ry(pi,0) # rotate by -pi around y-axis: q0:|-> in |+>
qc.ry(pi,1) # rotate by pi around y-axis: q1:|+> in |->
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# eigenvectors of Hadamard gate
# modify the sings in ini lists below to apply: --, ++, -+, +-
he1 = [1/sqrt(2), 1/sqrt(2)] #Q0
he2 = [1/sqrt(2), -1/sqrt(2)] #Q1
qc = QuantumCircuit(2)
qc.initialize(he1,0)  #q0 = |+>
qc.initialize(he2,1)  #q1 = |->
qc.ry(pi,1) # rotate by pi around y-axis: q1: |+> in |->
qc.cx(1,0)
qc.ry(-pi,1) # rotate by -pi around y-axis: q0: |+> in |->
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)

# WE CAN ALSO DO SWAP USING CLIFFORD GATES:
# if(state is |+>): Z|+>=|-> and Z|->=|+>.


#how to build any controlled rotation.
# consider arbitrary rotations around the y axis.
#  because the x and y axis are orthogonal, which causes the x gates to flip the direction of the rotation.
# CONTROLLED ROTATIONS
# start controlled pi around the y axis.
qc = QuantumCircuit(2)
c=0
t=1
qc.x(0)  # c is in |1>
qc.x(1)  # t is in |1>
theta = pi # theta can be anything (pi chosen arbitrarily)
qc.ry(theta/2,t) # rotate about y 90
qc.cx(c,t)  # if c is in |0> the rotations cancel out
            # if c is |1>: X-gate effecrt: flip direction of rotation, apply 2nd +pi/2.
            # bring it to the |0> pole
qc.ry(-theta/2,t) #brings it back from above or cancels 1st rot by 90
qc.cx(c,t) #again flips dir of previous rotation and apply +90
qc.draw() #effect: +pi rot about y-axis when c = |1>
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)
# CAN USE CNOT SIMILARLY TO MAKE
# CONTROLLED Z, and X. for angle theta.

# CONTROLLED VERSION OF ANY SINGLE QUBIT ROTATION V.
# FIRST NEED FIND 3 rotations A, B and C, and a phase ALPHA s.t.:
# ABC = I and e^ialpha(AZBZC) = v

# CONTROLLED VERSION OF ANY SINGLE-QUIBIT ROTATION, V.
# construct circuit.
# Gate(name, num_qubits, params, label=None)
A = Gate('A', 1, [])
B = Gate('B', 1, [])
C = Gate('C', 1, [])
angle = 1 # arbitrarily define alpha to allow drawing of circuit
c = 0
t = 1
qc = QuantumCircuit(2)
#append(instruction, qargs=None, cargs=None)
#Append one or more instructions to the end of the circuit,
#modifying the circuit in place. Expands qargs and cargs.
# ADD GATE C to ckt and feed t=1=q1 input to it.
qc.append(C, [t])
qc.cz(c,t) #controlled z-gate
# Append B to circuit...
qc.append(B, [t])
qc.cz(c,t)
# Same....
qc.append(A, [t])
qc.p(angle,c) # single qubit rot about z-axis. Z for 180, S for 90, T for 45.
qc.draw()

# circuit(q0,q1): q1-->C(cZ(q0,q1))B(cZ(q0,q1))A-->q1 = V

# THE TOFFOLI with 2 control: 1 target
# out is nand or and of controls depends on target in |1> or |0>
# applies X(t) if controls in |1>.
qc = QuantumCircuit(3)
a = 0
b = 1
t = 2
# Toffoli with control qubits a and b and target t
qc.ccx(a,b,t)
qc.draw()

# The following is an arbitrary (double control) U-gate applicable to one target qubit.
#
# U-gate = a gate which applied an induced phase to the target qubit iff both controls
# are in basis state |1>
#
# We define controlled versions of any rotation applied to one qubit:
# V = √U AND V-dagger and we implement using:
#
# cp = Controlled-Phase gate method, is a diagonal and symmetric gate that
# induces a phase on the state of the target qubit, depending on the control state.
a = 0
b = 1
t = 2
theta=pi/4
qc = QuantumCircuit(3)
qc.x(a)
qc.x(b)
qc.h(t)
qc.cp(theta,b,t) # our controlled-V = √U.
qc.cx(a,b)
qc.cp(-theta,b,t) # our controlled-V-dagger
qc.cx(a,b)
qc.cp(theta,a,t)
display(qc.draw())
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)


# ANOTHER WAY TO IMPLEMENT AND-GATE
qc = QuantumCircuit(3)
qc.ch(a,t)
qc.cz(b,t)
qc.ch(a,t)
qc.draw()


# due noise-induced gates, cannot have single qubit rotation about x, y, or z gates.
# Fault-tolerant schemes typically perform these rotations using multiple applications of
# H and T gates.
qc = QuantumCircuit(1)
qc.t(0) # T gate on qubit 0; rot on z by 45.
qc.draw()
# In the following we assume that the  H and  T gates are effectively perfect.
# This can be engineered by suitable methods for error correction and fault-tolerance.

# we use t gate to create similar rot about x axis.
qc = QuantumCircuit(1)
qc.h(0)
qc.t(0)
qc.h(0)
qc.draw()
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)



# R_z by 45 and R_x by 45:
#crucial property of the angle for this rotation is that it is an
#irrational multiple of  π
#everytime we apply a rotation that is larger than 2pi, we are doing an implicit mod by 2pi
#on the angle of rotation.
# cannot express irrational as simple fraction
#repeating combined rotation ntimes=>diff angle about same axis.
#if split angle into n intervals in [0,2pi] each of width=2pi/n
#each repetition, angle falls in one such slice
#for n+1 reps, at least 1 slice holds 2 angles.
#n1=reps for 1st angle
#n2=reps for 2nd angle
#angle for n2-n1 reps =θ12 s.t. θ12!=0 and is in [-2pi/n, 2pi/n]
# we can rotate around angles that are as small as we like by repeating the gate
# we can achive any angle of rotation in small parts with accurary up to 2pi/n.
# for arbitray rotations around one axis, we do Rz(pi/4) and Rx(pi/4) in opposite order.
qc = QuantumCircuit(1)
qc.h(0)
qc.t(0)
qc.h(0)
qc.t(0)
qc.draw()
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)

qc = QuantumCircuit(1)
qc.t(0)
qc.h(0)
qc.t(0)
qc.h(0)
qc.draw()
#have arbitrary rotation around two axes: to generate any arbitrary rotation around the Bloch sphere
#complexity of algorithms for fault-tolerant quantum computers: T gate count.
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)



# CLIFFOR SET: H, CNOT
#o. For every possible realization of fault-tolerant quantum computing, there is a set of quantum operations that are most straightforward to realize.

#these consist of single- and two-qubit gates, most of which correspond to the set of so-called Clifford gates. This is a very important set of operations, which do a lot of the heavy-lifting in any quantum algorithm.

#To understand Clifford gates, let's start with an example that you have already seen many times: the Hadamard.
# property of transforming Paulis into other Paulis is the defining feature of Clifford gates

# UPUdag'=P' or UPU'=P' for U in Cliffor and P,P' in Pauli

#For multiple-qubit Clifford gates, the defining property is that they transform tensor products of Paulis to other tensor products of Paulis. F
# sandwiching a matrix between a unitary and its Hermitian conjugate is known as conjugation by that unitary: process transforms the eigenstates of the matrix, but leaves the eigenvalues unchanged
#conjugation by Cliffords can transform between Paulis is because all Paulis share the same set of eigenvalues

#In order to do any quantum computation, we need gates that are not Cliffords
# e.g:  arbitrary rotations around the three axes of the qubit

#technique of boosting the power of non-Clifford gates by combining them with Clifford gates is one that we make great use of in quantum computing.

#e^ipi + 1 = 0. e^ix = cosx + isinx
# unitary and its corresponding Hermitian matrix have the same eigenstates.
# sandwiching a matrix between a unitary and its Hermitian conjugate is known as conjugation by that unitary.
# conjugation by a unitary transforms eigenstates and leaves eigenvalues unchanged



# CLIFFORD GATES: H, S, Paulis, 2qubit CNOT.
# The property has type in book: CNOT`•(X tensor_product I)•CNOT`= X tensor_product X
# Where: CNOT` = the rightmost CNOT matrix in section 3.1 of chapter 2.
# I is the 2 by 2 identity matrix and • is matrix multiplication as usually done.
# both sides equate to {{0, X},
#                        {X,0}}  which is a 4 by 4 when expanded.
#https://en.wikipedia.org/wiki/Clifford_gates

# COMPLEX CASE build from controlled-NOT gates
# X⊗X⊗X
qc = QuantumCircuit(3)
theta=90
qc.cx(0,2)
qc.cx(0,1)
qc.rx(theta,0)
qc.cx(0,1)
qc.cx(0,2)
display(qc.draw())
# Z⊗Z⊗Z
qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
qc.cx(0,2)
qc.cx(0,1)
qc.rx(theta,0)
qc.cx(0,1)
qc.cx(0,2)
qc.h(2)
qc.h(1)
qc.h(0)
display(qc.draw())

# when mapping from classical to quantum regime we need to account for
# REVERSIBILITY: We can have a unitary of the form U = SUMx |f(x)><x|
# and require 1:1 functions which is generally not true but for our purposes
# we can make true copying the input to the output... hence we get our
# Boolean oracle: U_f|x,0`> = |x, f(x)>
# With the computation expressed as a unitary we can consider all effects
# of appying it to superposition states.
# If we consider supersposition over all possible inputs x (not normalized)
# then, we get a supersposition over all possible I/O pairs: U_fSumx|x,0> = Sumx|x,f(x)>

# TO DISCARD INTERMEDIARY COMPUATIONS OR GARBAGE, WE DO THE FOLLOWING:

# consider algorithm computes:  V_f|x,0',0'> = |x, f(x), g(x)>
# g(x) = scratchpad register

# Quantum Algorithms typically built on intereference effects.
# SIMPLEST: create a supersposition using some unitary, then remove it with its inverse.
# result:  trivial, but such work is required... min work qc must do.

# EX: algorithm gives us supersposition state Sumx|x,f(x)>, we need to rbing it
# back to Sumx|x,0>. We can apply U_f_dag assuming circuit applied U_f (replace all gates with their inverse & reverse the order)
# If U_f not known...but we know V_f and hence need to apply V_f_dag.
# We need to remove garbage

# super simple example: verify trivial result: put in supersposition then undo it.
# Let x, f(x), g(x) consist of a single bit each such that f(x) = g(x) = x.
#****************************************************************************
# QUICK EXERCISES ASK: what if output register is initialized to |0> or |1>?
#. verify the output register and that result is only written there.
#****************************************************************************
svsim = Aer.get_backend('statevector_simulator')
input_bit = QuantumRegister(1, 'input')
output_bit = QuantumRegister(1, 'output')
garbage_bit = QuantumRegister(1, 'garbage')
# U_f
Uf = QuantumCircuit(input_bit, output_bit, garbage_bit)
#Uf.initialize([0,1],output_bit[0]) # ini only to |1> since state |0> is default.
Uf.cx(input_bit[0], output_bit[0])
print("Uf circuit:")
display(Uf.draw())
# we can see the collective state for initializing output to |0> or |1>
qobj = assemble(Uf)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Uf} = ", precision=1)
# V_f
Vf = QuantumCircuit(input_bit, output_bit, garbage_bit)
# cannot be initialize output here because it affects Vf.inverse computation.
# construct Vf with default values of all registers.
# Vf.x(input_bit[0]) # set input to 1.
Vf.cx(input_bit[0], garbage_bit[0])
Vf.cx(input_bit[0], output_bit[0])
print("Vf circuit")
display(Vf.draw())
qobj = assemble(Vf)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Vf} = ", precision=1)
# apply U_f then V_f_dagger
# when Qiskit computes the inverse, it will not accept Vf to have been initialized.
print("Vf^-1")
display(Vf.inverse().draw())
qc = Uf + Vf.inverse()  # leaves the first qubit entangled with unwanted garbage
#qc.initialize([1,0],1) # |0>
print("Uf + Vf^-1")
display(qc.draw())
# removing classical garbage from our quantum algorithms. A method known as 'uncomputation.
#We simply need to take another blank variable and apply V_f
final_output_bit = QuantumRegister(1, 'final-output')
copy = QuantumCircuit(output_bit, final_output_bit)
#copy.initialize([1,0],1) # |0>
copy.cx(output_bit, final_output_bit)
print("copy: output to final")
display(copy.draw())
qobj = assemble(copy)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{copy} = ", precision=1)
# The effect of this is to copies the information over ; it transforms the state.
# then, we apply V_f_dag which undoes the original computation.
# The net effect is to perform the computation without garbage, and we get our desired U_f
# whole process is this circuit:
final_circuit = (Vf.inverse() + copy + Vf)
# we now have all the tools we need to create quantum algorithms
print("Vf^-1 + copy + Vf:")
display(final_circuit.draw())
qobj = assemble(final_circuit)
final_state = svsim.run(qobj).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Final Circuit} = ", precision=1)



