
let x0 = input in
let x1 = op0 x0 in 
let x2 = op1 x1 in 
let x3 = op2 x2 in 
let output = x3
in output

~>

                              alloc x0[n0]
let x0 = input;               alloc x1[n1] in
let x1 = op0 x0; drop x0[n0]; alloc x2[n2] in 
let x2 = op1 x1; drop x1[n1]; alloc x3[n3] in 
let x3 = op2 x2; drop x2[n2]               in 
let output = x3; drop x3[n3]
in output



let x0 = input in
let x1 = op0 x0 in 
let x2 = op1 x1 in 
let x3 = op2 x2 in 
let x4 = add x1 x3 in
let output = x4
in output

~>

                                              alloc x0[n0]
let x0 = input;                               alloc x1[n1] in
let x1 = op0 x0;    drop x0[n0];              alloc x2[n2] in 
let x2 = op1 x1;                              alloc x3[n3] in 
let x3 = op2 x2;    drop x2[n2];              alloc x4[n4] in
let x4 = add x1 x3; drop x1[n1]; drop x3[n3];              in
let output = x4;    drop x4[n4]
in output
