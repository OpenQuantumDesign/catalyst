OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[1];

rx(0) q[2];
rx(1) q[3];
rx(2) q[3];
cx q[1],q[2];