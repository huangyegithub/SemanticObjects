grammar While;
/**
TODO: casts, unit, constraints on generics, drop special treatment of atomic types, FMU state copies
TODO LMOL: equations in LOAD statements
**/
@header {
package no.uio.microobject.antlr;
}
//Strings
STRING : '"' ('\\"'|.)*? '"' ;
VARSTRING : '?'LET+;

//Whitespace and comments
WS           : [ \t\r\n\u000C]+ -> channel(HIDDEN);
COMMENT      : '/*' .*? '*/' -> channel(HIDDEN) ;
LINE_COMMENT : '//' ~[\r\n]* -> channel(HIDDEN) ;

//Keywords: statements
SKIP_S : 'skip';
RETURN : 'return';
IF : 'if';
THEN : 'then';
NEW : 'new';
ELSE : 'else';
WHILE : 'while';
DO : 'do';
PRINTLN : 'print';
END : 'end';
ACCESS : 'access';
CONSTRUCT : 'construct';
MEMBER : 'member';
SIMULATE : 'simulate';
VALIDATE : 'validate';
TICK : 'tick';
BREAKPOINT : 'breakpoint';
SUPER : 'super';
DESTROY : 'destroy';
LOAD : 'load';
ABSTRACT : 'abstract'; //'abstract' collides with Java

//Keywords: classes and methods
CLASS : 'class';
EXTENDS : 'extends';
RULE : 'rule';
OVERRIDE : 'override';
MAIN : 'main';
PRIVATE : 'private';
PROTECTED : 'protected';
INFERPRIVATE : 'nonsemantic';
MODELS : 'models';
DOMAIN : 'domain';
ANCHOR : 'anchor';
RETRIEVE : 'retrieve';
RETRIEVABLE : 'link';
BACKWARDS : 'back';

//Keywords: constants
TRUE : 'True';
FALSE : 'False';
NULL : 'null';
THIS: 'this';
UNIT: 'unit';

//Keywords: operators
EQ : '=';
NEQ : '<>';
LT : '<';
GT : '>';
LEQ : '<=';
GEQ : '>=';
ASS : ':=';
PLUS : '+';
MULT : '*';
MINUS : '-';
DIV : '/';
MOD : '%';
AND : '&';
OR : '|';
NOT : '!';

//Keywords: others
DOT : '.';
SEMI : ';';
OPARAN : '(';
CPARAN : ')';
OBRACK : '[';
CBRACK : ']';
COMMA : ',';
FMU : 'Cont';
PORT : 'port';
SPARQLMODE : 'SPARQL';
INFLUXMODE : 'INFLUXDB';

//Names etc.
fragment DIG : [0-9];
fragment LET : [a-zA-Z_];
fragment LOD : LET | DIG;
NAME : LET LOD*;
CONSTANT :  DIG+;
FLOAT : DIG* DOT DIG*;

namelist : NAME (COMMA NAME)*;

//Entry point
program : (class_def)* MAIN statement END;

//classes
class_def : (abs=ABSTRACT)? CLASS  className = NAME (LT namelist GT)? (EXTENDS superType = type)? (ANCHOR anchorVar=VARSTRING)? OPARAN fieldDeclList? CPARAN
            (models_block)?
            method_def*
            END (RETRIEVE retrieveQuery = STRING)?;
method_def :  (abs=ABSTRACT)? (visibility=visibilitymodifier)? (builtinrule=RULE)? (domainrule=DOMAIN)? (overriding=OVERRIDE)? type NAME OPARAN paramList? CPARAN (statement END)?;

models_block : MODELS owldescription=STRING SEMI                                                    #simple_models_block
             | MODELS OPARAN guard=expression CPARAN owldescription=STRING SEMI models_block        #complex_models_block
             ;
//Statements
statement :   SKIP_S SEMI                                                                                                                               # skip_statment
			| (declType = type)? expression ASS expression SEMI                                                                                         # assign_statement
			| ((declType = type)? target=expression ASS)? SUPER OPARAN (expression (COMMA expression)*)? CPARAN SEMI                                    # super_statement
			| RETURN expression SEMI                                                                                                                    # return_statement
			| fmu=expression DOT TICK OPARAN time=expression CPARAN SEMI                                                                                # tick_statement
			| ((declType = type)? target=expression ASS)? expression DOT NAME OPARAN (expression (COMMA expression)*)? CPARAN SEMI                      # call_statement
			| (declType = type)? target=expression ASS NEW newType = type OPARAN (expression (COMMA expression)*)? CPARAN (MODELS owldescription = expression)? SEMI                          # create_statement
			| BREAKPOINT SEMI                                                                                                                           # debug_statement
			| PRINTLN OPARAN expression CPARAN SEMI                                                                                                     # output_statement
			| DESTROY OPARAN expression CPARAN SEMI                                                                                                     # destroy_statement
			| (declType = type)? target=expression ASS ACCESS OPARAN query=expression (COMMA lang=modeexpression)? (COMMA expression (COMMA expression)*)? CPARAN SEMI               # sparql_statement
			| (declType = type)? target=expression ASS CONSTRUCT OPARAN query=expression (COMMA expression (COMMA expression)*)? CPARAN SEMI            # construct_statement
			| (declType = type)? target=expression ASS MEMBER OPARAN query=expression CPARAN SEMI                                                       # owl_statement
			| (declType = type)? target=expression ASS VALIDATE OPARAN query=expression CPARAN SEMI                                                     # validate_statement
			| (declType = type)? target=expression ASS SIMULATE OPARAN path=STRING (COMMA varInitList)? CPARAN SEMI                                     # simulate_statement
			| IF expression THEN thenS=statement (ELSE elseE=statement)? END next=statement?                                                            # if_statement
            | WHILE expression DO statement END next=statement?                                                                                         # while_statement
            | (declType = type)? target=expression ASS LOAD newType = type OPARAN CPARAN SEMI                                                           # retrieve_statement
            | statement statement                                                                                                                       # sequence_statement
            ;

modeexpression : SPARQLMODE                         #sparql_mode
               | INFLUXMODE OPARAN STRING CPARAN    #influx_mode
               ;
//Expressions
expression :      THIS                           # this_expression
                | THIS DOT NAME                  # field_expression
                | NAME                           # var_expression
                | CONSTANT                       # const_expression
                | TRUE                           # true_expression
                | FALSE                          # false_expression
                | STRING                         # string_expression
                | FLOAT                          # double_expression
                | NULL                           # null_expression
                | UNIT                           # unit_expression
                | expression DOT PORT OPARAN STRING CPARAN	# fmu_field_expression
                | expression DOT NAME			 # external_field_expression
                | expression DIV expression      # div_expression
                | expression MOD expression      # mod_expression
                | expression MULT expression     # mult_expression
                | expression PLUS expression     # plus_expression
                | expression MINUS expression    # minus_expression
                | expression EQ expression       # eq_expression
                | expression NEQ expression      # neq_expression
                | expression GEQ expression      # geq_expression
                | expression LEQ expression      # leq_expression
                | expression GT expression       # gt_expression
                | expression LT expression       # lt_expression
                | expression AND expression      # and_expression
                | expression OR expression       # or_expression
                | NOT expression                 # not_expression
                | OPARAN expression CPARAN       # nested_expression
                ;

type : NAME                                                    #simple_type
     | NAME LT typelist GT                                     #nested_type
     | FMU OBRACK in=paramList? SEMI out=paramList? CBRACK     #fmu_type
     ;
typelist : type (COMMA type)*;
param : type NAME;
paramList : param (COMMA param)*;
fieldDecl : (infer=INFERPRIVATE)? (visibility=visibilitymodifier)? (domain=DOMAIN)? ((backwards=BACKWARDS)? RETRIEVABLE OPARAN query=STRING CPARAN)? type NAME;
fieldDeclList : fieldDecl (COMMA fieldDecl)*;
varInit : NAME ASS expression;
varInitList : varInit (COMMA varInit)*;

visibilitymodifier : PRIVATE | PROTECTED;
