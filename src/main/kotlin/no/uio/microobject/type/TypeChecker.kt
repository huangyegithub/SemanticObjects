package no.uio.microobject.type

import no.uio.microobject.antlr.WhileParser
import no.uio.microobject.data.TripleManager
import no.uio.microobject.main.Settings
import no.uio.microobject.runtime.FieldInfo
import no.uio.microobject.runtime.SimulatorObject
import no.uio.microobject.runtime.Visibility
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.RuleContext
import org.javafmi.wrapper.Simulation
import java.nio.file.Files
import java.nio.file.Paths


/**
 *
 * This has for now the following quirks:
 *  - Generic arguments share a global namespace
 *  - Variables cannot be declared twice in a method, even if the scopes do not overlap
 *  - Generic classes cannot be extended
 *
 */

class TypeChecker(private val ctx: WhileParser.ProgramContext, private val settings: Settings, private val tripleManager: TripleManager) : TypeErrorLogger() {

    companion object{
        /**********************************************************************
        Handling type data structures
         ***********************************************************************/
        //translates a string text to a type, even accessed within class createClass (needed to determine generics)
        private fun stringToType(text: String, createClass: String, generics : MutableMap<String, List<String>>) : Type {
            return when {
                //generics.getOrDefault(createClass, listOf()).contains(text) -> GenericType(text)
                generics.values.flatten().contains(text) -> GenericType(text)
                text == INTTYPE.name -> INTTYPE
                text == BOOLEANTYPE.name -> BOOLEANTYPE
                text == STRINGTYPE.name -> STRINGTYPE
                text == DOUBLETYPE.name -> DOUBLETYPE
                text == UNITTYPE.name -> UNITTYPE
                else -> BaseType(text)
            }
        }

        //translates a type AST text to a type, even accessed within class createClass (needed to determine generics)
        fun translateType(ctx : WhileParser.TypeContext, className : String, generics : MutableMap<String, List<String>>) : Type {
            return when(ctx){
                is WhileParser.Simple_typeContext -> stringToType(ctx.text, className, generics)
                is WhileParser.Nested_typeContext -> {
                    val lead = stringToType(ctx.NAME().text, className, generics)
                    ComposedType(lead, ctx.typelist().type().map { translateType(it, className, generics) })
                }
                is WhileParser.Fmu_typeContext -> {
                    // Here we compare against literal strings "in", "out"; if
                    // the grammar is changed, adapt the two `filter`
                    // expressions below accordingly.
                    val ins = if(ctx.fmuParamList() != null) ctx.fmuParamList().fmuparam().filter { it.direction.text == "in" }.map { Pair(it.param().NAME().text, translateType(it.param().type(), className, generics)) }
                    else emptyList()
                    val outs = if(ctx.fmuParamList() != null) ctx.fmuParamList().fmuparam().filter { it.direction.text == "out" }.map { Pair(it.param().NAME().text, translateType(it.param().type(), className, generics)) }
                    else emptyList()
                    SimulatorType(ins,outs)
                }
                is WhileParser.Scenario_typeContext -> ScenarioType
                else -> throw Exception("Unknown type context: $ctx") // making the type checker happy
            }
        }
    }

    //Known classes
    private val classes : MutableSet<String> = mutableSetOf("Int", "Boolean", "Unit", "String", "Object", "Double")

    //Type hierarchy. Null is handled as a special case during the analysis itself.
    private val extends : MutableMap<String, Type> = mutableMapOf(Pair("Double", OBJECTTYPE), Pair("Int", OBJECTTYPE), Pair("Boolean", OBJECTTYPE), Pair("Unit", OBJECTTYPE), Pair("String", OBJECTTYPE))

    //List of declared generic type names per class
    private val generics : MutableMap<String, List<String>> = mutableMapOf()

    //List of declared fields (with type and visibility) per class
    private val fields : MutableMap<String, Map<String, FieldInfo>> = mutableMapOf()


    //List of declared methods per class
    private val methods : MutableMap<String, List<WhileParser.Method_defContext>> = mutableMapOf()

    //Easy access by mapping each class name to its definition
    private val recoverDef :  MutableMap<String, WhileParser.Class_defContext> = mutableMapOf()


    /**********************************************************************
    ISSA
     ***********************************************************************/
    internal val queryCheckers = mutableListOf<QueryChecker>()

    override fun report(silent: Boolean): Boolean {
        return queryCheckers.fold(super.report(silent)) { acc, nx -> acc && nx.report(silent) }
    }


    /**********************************************************************
    Preprocessing
     ***********************************************************************/
    internal fun collect(){
        //three passes for correct translation of types
        for(clCtx in ctx.class_def()){
            val name = clCtx.className.text
            classes.add(name)
            recoverDef[name] = clCtx
        }
        for(clCtx in ctx.class_def()){ // do extra pass to correctly compute extends
            val name = clCtx.className.text
            if(clCtx.namelist() != null)
                generics[name] = clCtx.namelist().NAME().map { it.text }
        }
        for(clCtx in ctx.class_def()){
            val name = clCtx.className.text
            if(clCtx.superType != null)  extends[name] = translateType(clCtx.superType, name, generics)
            else                         extends[name] = OBJECTTYPE

            methods[name] = computeMethods(name)
            fields[name] = computeFields(name).associate {
                val cVisibility =
                    if (it.HIDE() == null) Visibility.HIDE else Visibility.DEFAULT
                Pair(
                    it.NAME().text,
                    FieldInfo(
                        it.NAME().text,
                        translateType(it.type(), name, generics),
                        cVisibility,
                        BaseType(name),
                        it.domain != null
                    )
                )
            }
        }
    }

    private fun computeMethods(className: String) : List<WhileParser.Method_defContext>{
        val current = ctx.class_def().firstOrNull { it.className.text == className } ?: return emptyList()
        val above = if(current.superType != null) computeMethods(current.superType.text) else emptyList()
        val own = if(current.method_def() != null) current.method_def() else emptyList()
        return above + own
    }

    private fun computeFields(className: String) : List<WhileParser.FieldDeclContext>{
        val current = ctx.class_def().firstOrNull { it.className.text == className } ?: return emptyList()
        val above = if(current.superType != null) computeFields(current.superType.text) else emptyList()
        val own = if(current.fieldDeclList() != null) current.fieldDeclList().fieldDecl() else emptyList()
        return above + own
    }

    /**********************************************************************
    Actual type checking
     ***********************************************************************/

    /* interface */
    fun check() {
        //preprocessing
        collect()

        //check classes
        for (clCtx in ctx.class_def()) checkClass(clCtx)

        //check main block
        checkStatement(ctx.statement(), false, mutableMapOf(), ERRORTYPE, ERRORTYPE, ERRORTYPE.name, false)
    }

    internal fun checkClass(clCtx : WhileParser.Class_defContext){
        val name = clCtx.className.text


        //Check extends: class must exist and not *also* be generic
        if (clCtx.superType != null) {
            val superType = translateType(clCtx.superType, name, generics)
            val extName = superType.getPrimary().getNameString()
            if (!classes.contains(extName)) log("Class $name extends unknown class $extName.", clCtx)
            else {
                if(recoverDef.containsKey(extName) && recoverDef[extName]!!.namelist() != null && recoverDef[name]!!.namelist() != null)
                    log("Generic class $name extends generic class $extName.", clCtx)
                if(!superType.isFullyConcrete())
                    log("Class $name extends generic class $extName but does not instantiate all generics.", clCtx)
                if(superType is ComposedType && superType.params.size != recoverDef[extName]!!.namelist().NAME().size)
                    log("Class $name extends generic class $extName but does not instantiate all generics.", clCtx)
                if(recoverDef[extName]!!.namelist() != null && superType !is ComposedType )
                    log("Class $name extends generic class $extName but does not instantiate all generics.", clCtx)
            }
        }

        //Check generics: no shadowing
        if(clCtx.namelist() != null){
            val collisions = clCtx.namelist().NAME().filter { classes.contains(it.text)  }
            for(col in collisions)
                log("Class $name has type parameter ${col.text} that shadows a known type.", clCtx)

            val globalColl = clCtx.namelist().NAME().filter { it -> generics.filter { it.key != name }.values.fold(listOf<String>()) { acc, nx -> acc + nx }
                .contains(it.text) }
            for(col in globalColl)
                log("Class $name has type parameter ${col.text} that shadows another type parameter (type parameters share a global namespace).", clCtx)
        }

        //Check fields
        if(clCtx.fieldDeclList() != null){
            for( param in clCtx.fieldDeclList().fieldDecl()){
                val paramName = param.NAME().text
                val paramType = translateType(param.type(), name, generics)
                if(containsUnknown(paramType, classes))
                    log("Class $name has unknown type $paramType for field $paramName.", param)
                if(param.domain != null && paramType != INTTYPE && paramType != BOOLEANTYPE && paramType != STRINGTYPE )
                    log("Domain fields must be literal types, but $paramType found", param)
            }
        }

        //Check methods
        if(clCtx.method_def() != null)
            for( mtCtx in clCtx.method_def() ) checkMet(mtCtx, name)

        //Check abstract
        if(clCtx.abs == null) {
            val leftOver = getLeftoverAbstract(clCtx)
            if (leftOver.isNotEmpty())
                log(
                    "Class $name does not implement the following abstract methods: $leftOver",
                    clCtx
                )
        }
        //Only booleans in model block
        if(clCtx.models_block() != null) checkModels(clCtx.models_block(), fields[name]!!, BaseType(name))

    }

    private fun checkModels(modelsBlock: WhileParser.Models_blockContext,
                            fields : Map<String, FieldInfo>,
                            thisType : Type) {
        if(modelsBlock is WhileParser.Complex_models_blockContext){
            val t = getType(modelsBlock.guard, fields, emptyMap(), thisType, false)
            if(t != BOOLEANTYPE) log("Models guards must be booleans over the fields", modelsBlock)
            checkModels(modelsBlock.models_block(), fields, thisType)
        }
    }

    private fun checkOverride(mtCtx: WhileParser.Method_defContext, className: String){
        val name = mtCtx.NAME().text
        val upwards = extends.getOrDefault(className, ERRORTYPE).getPrimary().getNameString()
        val allMets = getMethods(upwards).filter { it.NAME().text == name }
        val cDecl = getClassDecl(mtCtx)
        val match = if(cDecl != null) getGenericAssignment(cDecl) else emptyMap()


        if (allMets.isEmpty()) {
            log(
                "Method $name is declared as overriding in $className, but no superclass of $className implements $name.",
                mtCtx,
                Severity.WARNING
            )
        }
        for (superMet in allMets) {
            val beforeMatching = translateType(superMet.type(), className, generics)
            val extracted = generics.map { it.value }.flatten()
            val afterMatching = applyMatching(beforeMatching, match, extracted)
            val transType = translateType(mtCtx.type(), className, generics)
            if ( afterMatching != transType )
                log(
                    "Method $name is declared as overriding in $className, but a superclass has a different return type: ${
                        translateType(
                            superMet.type(),
                            className
                            , generics)
                    }.", mtCtx
                )
            if(mtCtx.paramList() != null) {
                if (superMet.paramList() != null) {
                    if (superMet.paramList().param().size != mtCtx.paramList().param().size)
                        log(
                            "Method $name is declared as overriding in $className, but a superclass has a different number of parameters: ${
                                superMet.paramList().param().size
                            }.", mtCtx
                        )
                    else {
                        for (i in mtCtx.paramList().param().indices) {
                            val myName = mtCtx.paramList().param(i).NAME().text
                            val otherName = superMet.paramList().param(i).NAME().text
                            val myType = translateType(mtCtx.paramList().param(i).type(), className, generics)
                            val otherType = translateType(superMet.paramList().param(i).type(), className, generics)

                            val beforeMatching = translateType(superMet.paramList().param(i).type(), className, generics)
                            val extracted = generics.map { it.value }.flatten()
                            val afterMatching = applyMatching(beforeMatching, match, extracted)
                            if (myName != otherName)
                                log(
                                    "Method $name is declared as overriding in $className, but parameter $i ($myType $myName) has a different name in a superclass of $className: $otherName.",
                                    mtCtx,
                                    Severity.WARNING
                                )
                            if (myType != afterMatching)
                                log(
                                    "Method $name is declared as overriding in $className, but parameter $i ($myType $myName) has a different type in a superclass of $className: $otherType.",
                                    mtCtx
                                )
                        }
                    }
                } else log("Method $name is declared as overriding in $className, but a superclass has a different number of parameters.", mtCtx)
            } else if(superMet.paramList() != null)
                log("Method $name is declared as overriding in $className, but a superclass has a different number of parameters.", mtCtx)
        }
    }

    internal fun checkMet(mtCtx: WhileParser.Method_defContext, className: String) {
        val name = mtCtx.NAME().text
        if(name == "port")
            log("Methods are not allows to be called \"port\".", mtCtx)
        val gens = generics.getOrDefault(className, listOf()).map { GenericType(it) }
        val thisType = if (gens.isNotEmpty()) ComposedType(BaseType(className), gens) else BaseType(className)
        val retType = translateType(mtCtx.type(), className, generics)
        val cDef = recoverDef[className]
        if(cDef != null){
            if (mtCtx.abs != null && cDef.abs == null)
                log("Abstract methods are only allowed within abstract classes", mtCtx)
        }
        if(mtCtx.abs != null && mtCtx.builtinrule != null)
            log("Rule methods are not allowed to be abstract", mtCtx)
        if(mtCtx.abs != null && mtCtx.overriding != null)
            log("Overriding methods are not allowed to be abstract", mtCtx)


        //Check rule annotation
        if(mtCtx.builtinrule != null && mtCtx.paramList() != null)
            log("rule-method $className.$name has non-empty parameter list.", mtCtx)

        //Check return type: must be known
        if(containsUnknown(retType, classes))
            log("Method $className.$name has unknown return type ${mtCtx.type()}.", mtCtx.type())

        //Check parameters: no shadowing, types must be known
        if(mtCtx.paramList() != null){
            for(param in mtCtx.paramList().param()){
                val paramType = translateType(param.type(), className, generics)
                val paramName = param.NAME().text
                if(containsUnknown(paramType, classes))
                    log("Method $className.$name has unknown parameter type $paramType for parameter $paramName.", mtCtx.type())

                if(fields[className]!!.containsKey(paramName))
                    log("Method $className.$name has parameter $paramName that shadows a field.", mtCtx)
            }
        }


        if(mtCtx.abs != null) //
            return

        //Check overriding
        if(mtCtx.overriding != null){
            if(recoverDef[className]!!.superType != null) {
                checkOverride(mtCtx, className)
            } else {
                log("Method $name is declared as overriding in $className, but $className does not extend any other class.", mtCtx, Severity.WARNING)
            }
       }
        if(mtCtx.overriding == null && recoverDef[className]!!.superType != null){
            val upwards = extends.getOrDefault(className, ERRORTYPE).getPrimary().getNameString()
            val allMets = getMethods(upwards).filter { it.NAME().text == name }
            if(allMets.isNotEmpty()){
                log("Method $name is not declared as overriding in $className, but a superclass of $className implements $name.", mtCtx, Severity.WARNING)
            }
        }

        //Check statement
        val initVars = if(mtCtx.paramList() != null) mtCtx.paramList().param()
            .associate { Pair(it.NAME().text, translateType(it.type(), className, generics)) }.toMutableMap() else mutableMapOf()
        val ret = checkStatement(mtCtx.statement(), false, initVars, translateType(mtCtx.type(), className, generics), thisType, className, mtCtx.builtinrule != null)

        if(!ret && retType != UNITTYPE) {
            log("Method ${mtCtx.NAME().text} has non-Unit return type a path without a final return statement.", mtCtx)
        }

        if(mtCtx.domainrule != null && retType != INTTYPE && retType != BOOLEANTYPE && retType != STRINGTYPE )
            log("Domain metnhod must have literal retur types, but $retType found", mtCtx)

        //check queries
        if(settings.useQueryType) queryCheckers.forEach { it.type(tripleManager) }
    }

    // This cannot be done with extension methods because they cannot override StatementContext.checkStatement()
    // TODO: move this to an antlr visitor
    private fun checkStatement(ctx : WhileParser.StatementContext,
                               finished : Boolean,
                               vars : MutableMap<String, Type>,
                               metType : Type, //return type
                               thisType : Type,
                               className: String,
                               inRule : Boolean ) : Boolean {
        val inner : Map<String, FieldInfo> = getFields(className)

        when(ctx){
            is WhileParser.If_statementContext -> {
                val innerType = getType(ctx.expression(), inner, vars, thisType, inRule)
                if(innerType != ERRORTYPE && innerType != BOOLEANTYPE)
                    log("If statement expects a boolean in its guard, but parameter has type $innerType.",ctx)
                val left = checkStatement(ctx.statement(0), finished, vars, metType, thisType, className, inRule)
                val right = if(ctx.ELSE() != null) checkStatement(ctx.statement(1), finished, vars, metType, thisType, className, inRule) else true
                return if(ctx.next != null){
                    checkStatement(ctx.next,  (left && right) || finished, vars, metType, thisType, className, inRule)
                } else (left && right) || finished
            }
            is WhileParser.While_statementContext -> {
                val innerType = getType(ctx.expression(), inner, vars, thisType, inRule)
                if(innerType != ERRORTYPE && innerType != BOOLEANTYPE)
                    log("While statement expects a boolean in its guard, but parameter has type $innerType.",ctx)
                checkStatement(ctx.statement(0), finished, vars, metType, thisType, className, inRule)
                if(ctx.next != null)
                    return checkStatement(ctx.next,  finished, vars, metType, thisType, className, inRule)
            }
            is WhileParser.Sequence_statementContext -> {
                val first = checkStatement(ctx.statement(0), finished, vars, metType, thisType, className, inRule)
                return checkStatement(ctx.statement(1), first, vars, metType, thisType, className, inRule)
            }
            is WhileParser.Assign_statementContext -> {
                val lhsType =
                    if(ctx.type() != null){
                        val lhs = ctx.expression(0)
                        if(lhs !is WhileParser.Var_expressionContext){
                            log("Variable declaration must declare a variable.", ctx)
                        } else {
                            val name = lhs.NAME().text
                            if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                            else vars[name] = translateType(ctx.type(), className, generics)
                        }
                        translateType(ctx.type(), className, generics)
                    } else getType(ctx.expression(0), inner, vars, thisType, false, read = false)
                val rhsType = getType(ctx.expression(1), inner, vars, thisType, inRule)
                if(lhsType != ERRORTYPE && rhsType != ERRORTYPE && !lhsType.isAssignable(rhsType, extends))
                    log("Type $rhsType is not assignable to $lhsType", ctx)
                if(ctx.expression(0) !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Super_statementContext -> {
                val lhsType =
                    when {
                        ctx.type() != null -> {
                            val lhs = ctx.expression(0)
                            if (lhs !is WhileParser.Var_expressionContext) {
                                log("Variable declaration must declare a variable.", ctx)
                            } else {
                                val name = lhs.NAME().text
                                if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                                else vars[name] = translateType(ctx.type(), className, generics)
                            }
                            translateType(ctx.type(), className, generics)
                        }
                        ctx.target != null -> {
                            getType(ctx.expression(0), inner, vars, thisType, inRule, read = false)
                        }
                        else -> {
                            null
                        }
                    }

                if (lhsType != null && !lhsType.isAssignable(metType, extends))
                    log("Return value of super call cannot be assigned to type.", ctx)


                var metCtx: RuleContext? = ctx
                while (metCtx != null && metCtx !is WhileParser.Method_defContext) {
                    metCtx = metCtx.parent
                }
                val methodName = (metCtx as WhileParser.Method_defContext).NAME().text

                val calleeIndex = if (ctx.target == null) 0 else 1
                val met = methods.getOrDefault(className, listOf()).first { it.NAME().text == methodName }
                if (ctx.expression() != null) {
                    val callParams: List<Type> = getParameterTypes(met, className)
                    if (ctx.expression().size - calleeIndex != callParams.size) {
                        log(
                            "Mismatching number of parameters when calling super in $methodName. Expected ${
                                callParams.size
                            }, got ${ctx.expression().size  - calleeIndex}", ctx
                        )
                    } else {
                        for (i in calleeIndex until ctx.expression().size) {
                            val match = i - calleeIndex
                            val targetType = callParams[match] //of method decl
                            val realType =
                                getType(ctx.expression(i), inner, vars, thisType, inRule)                    //of call
                            if (targetType != ERRORTYPE && realType != ERRORTYPE && !realType.isAssignable(targetType, extends))
                                log("Type $realType is not assignable to $targetType.", ctx)

                        }
                    }
                }

                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Call_statementContext -> {
                val lhsType =
                    when {
                        ctx.type() != null -> {
                            val lhs = ctx.expression(0)
                            if(lhs !is WhileParser.Var_expressionContext){
                                log("Variable declaration must declare a variable.", ctx)
                            } else {
                                val name = lhs.NAME().text
                                if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                                else vars[name] = translateType(ctx.type(), className, generics)
                            }
                            translateType(ctx.type(), className, generics)
                        }
                        ctx.target != null -> {
                            getType(ctx.expression(0), inner, vars, thisType, false, read = false)
                        }
                        else -> {
                            null
                        }
                    }

                val calleeIndex = if(ctx.target == null) 0 else 1
                val rhsType = getType(ctx.expression(calleeIndex), inner, vars, thisType, inRule, inRule)
                if(rhsType != ScenarioType && rhsType !is SimulatorType && (rhsType.getPrimary() !is BaseType || !methods.containsKey(rhsType.getPrimary().getNameString()))){
                    log("Call on type $rhsType not possible: type $rhsType is not a class and not a scenario.", ctx)
                } else {
                  val calledMet = ctx.NAME().text
                  val nName = rhsType.getPrimary().getNameString()
                  if(!methods.getOrDefault(nName, listOf()).map { it.NAME().text }.contains(calledMet)){
                      if(rhsType == ScenarioType){
                          if(calledMet == "check"){
                              if(ctx.expression().size - 1 - calleeIndex != 0)
                                  log("The scenario.check method takes no parameter.", ctx)
                              if(lhsType != null && lhsType != BOOLEANTYPE)
                                  log("Type Boolean is not assignable to $lhsType.", ctx)
                          } else if (calledMet == "assign"){
                              if(lhsType != null)
                                  log("The scenario.assign method returns no value.", ctx)
                              if(ctx.expression().size - 1 - calleeIndex != 1)
                                  log("The scenario.assign method requires one parameter.", ctx)
                              else{
                                  val callType = getType(ctx.expression(1), inner, vars, thisType, inRule)
                                  if(callType !is SimulatorType)
                                      log("The scenario.assign method requires a FMO[...] parameter, got $callType.", ctx)
                              }
                          } else {
                              log("Call on type $rhsType not possible: method $calledMet not found.", ctx)
                          }
                      } else if (rhsType is SimulatorType) {
                          if(calledMet == "canTick"){
                              if(ctx.expression().size - 1 - calleeIndex != 0)
                                  log("The FMO.$calledMet method takes no parameter.", ctx)
                              if(lhsType != null && lhsType != BOOLEANTYPE)
                                  log("Type Boolean is not assignable to $lhsType.", ctx)
                          } else if(calledMet.startsWith("canSet_")) {
                              if(ctx.expression().size - 1 - calleeIndex != 0)
                                  log("The FMO.$calledMet method takes no parameter.", ctx)
                              if(lhsType != null && lhsType != BOOLEANTYPE)
                                  log("Type Boolean is not assignable to $lhsType.", ctx)
                              val field = calledMet.substring(7)
                              if(rhsType.inVar.none { it.first == field })
                                  log("Field $field is not an input variable of $rhsType, cannot call $calledMet.", ctx)
                          } else if(calledMet.startsWith("canGet_")) {
                              if(ctx.expression().size - 1 - calleeIndex != 0)
                                  log("The FMO.$calledMet method takes no parameter.", ctx)
                              if(lhsType != null && lhsType != BOOLEANTYPE)
                                  log("Type Boolean is not assignable to $lhsType.", ctx)
                              val field = calledMet.substring(7)
                              if(rhsType.outVar.none { it.first == field })
                                  log("Field $field is not an output variable of $rhsType, cannot call $calledMet.", ctx)
                          } else {
                              log("Call on type $rhsType not possible: method $calledMet not found.", ctx)
                          }
                      } else {
                          log("Call on type $rhsType not possible: method $calledMet not found.", ctx)
                      }
                  } else {
                      val otherClassName = rhsType.getPrimary().getNameString()
                      val met = methods.getOrDefault(otherClassName, listOf()).first { it.NAME().text == calledMet }
                      if(met.builtinrule == null && inRule)
                          log("Method $otherClassName.$calledMet is not a rule-method, but accessed from one.", ctx)
                      if(met.paramList() != null) {
                          val callParams : List<Type> = getParameterTypes(met, otherClassName)
                          if (ctx.expression().size - 1 - calleeIndex != callParams.size) {
                              log(
                                  "Mismatching number of parameters when calling $rhsType.$calledMet. Expected ${
                                      callParams.size
                                  }, got ${ctx.expression().size - 1 - calleeIndex}", ctx
                              )
                          } else {
                              for (i in calleeIndex + 1 until ctx.expression().size) {
                                  val match = i - calleeIndex - 1
                                  val targetType = callParams[match] //of method decl
                                  val realType = getType(ctx.expression(i), inner, vars, thisType, inRule)                    //of call
                                  val finalType = instantiateGenerics(targetType, rhsType, otherClassName, generics.getOrDefault(className, listOf()))
                                  if (targetType != ERRORTYPE && realType != ERRORTYPE && !finalType.isAssignable(realType, extends)) {
                                      log("Type $realType is not assignable to $targetType.", ctx)
                                  }
                              }
                              if (lhsType != null) { //result type
                                  val metRet = translateType(met.type(), otherClassName, generics) // type as declared
                                  val finalType = instantiateGenerics(metRet, rhsType, otherClassName, generics.getOrDefault(className, listOf()))
                                  if (!lhsType.isAssignable(finalType, extends)) {
                                      log(
                                          "Type $finalType is not assignable to $lhsType.",
                                          ctx
                                      )
                                  }
                              }
                          }
                      }
                  }
                }
                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Create_statementContext -> {
                val lhsType =
                    if (ctx.declType != null) {
                        val lhs = ctx.expression(0)
                        if(lhs !is WhileParser.Var_expressionContext){
                            log("Variable declaration must declare a variable.", ctx)
                        } else {
                            val name = lhs.NAME().text
                            if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                            else vars[name] = translateType(ctx.declType, className, generics)
                        }
                        translateType(ctx.declType, className, generics)
                    } else getType(ctx.expression(0), inner, vars, thisType, false, read = false)
                val newTypeFound =
                    translateType(ctx.newType, className, generics)
                val createClass = newTypeFound.getPrimary().getNameString()
                val createDecl = recoverDef[createClass]
                if(createDecl!!.abs != null)
                    log("Cannot instantiate abstract class $createClass", ctx)
                val newType : Type = newTypeFound

                /*if(ctx.namelist() != null)
                    newType = ComposedType(newType, ctx.namelist().NAME().map {
                        stringToType(it.text, className, generics)
                    })

                if(createDecl?.namelist() != null){
                    if(ctx.namelist() == null ){
                        log("Generic parameters for $createClass missing.", ctx)
                    }else if(createDecl.namelist().NAME().size != ctx.namelist().NAME().size){
                        log("Number of generic parameters for $createClass is wrong.", ctx)
                    }
                }*/

                val creationParameters = getParameterTypes(createClass)
                if (creationParameters.size == (ctx.expression().size - (if(ctx.owldescription == null) 1 else 2))){
                    for(i in 1 until creationParameters.size+1){
                        if(ctx.expression() == ctx.owldescription) continue
                        val targetType = creationParameters[i-1]
                        val finalType = instantiateGenerics(targetType, newType, createClass, generics.getOrDefault(className, listOf()))
                        val realType = getType(ctx.expression(i), inner, vars, thisType, inRule)
                        if(targetType != ERRORTYPE && realType != ERRORTYPE && !finalType.isAssignable(realType, extends)) {
                            log("Type $realType is not assignable to $finalType", ctx)
                        }
                    }
                } else {
                    log("Mismatching number of parameters when creating an $createClass instance. Expected ${creationParameters.size}, got ${ctx.expression().size-1}", ctx)
                }

                if(lhsType != ERRORTYPE && !lhsType.isAssignable(newType, extends) ) {
                    log("Type $newType is not assignable to $lhsType", ctx)
                }

                if(ctx.owldescription != null && getType(ctx.owldescription, inner, vars, thisType, false) != STRINGTYPE){
                    log("Models clause must be a String: ${ctx.owldescription}", ctx)
                }

                if(inRule) log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Sparql_statementContext -> {
                if(ctx.lang is WhileParser.Influx_modeContext){
                    log("Flux queries are not supported for type checking yet", ctx, Severity.WARNING)
                }else {
                    var expType: Type? = null
                    if (ctx.declType != null) {
                        val lhs = ctx.expression(0)
                        if (lhs !is WhileParser.Var_expressionContext) {
                            log("Variable declaration must declare a variable.", ctx)
                        } else {
                            val name = lhs.NAME().text
                            if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                            else {
                                expType = translateType(ctx.type(), className, generics)
                                vars[name] = expType
                            }
                        }
                    } else {
                        expType = getType(ctx.target, inner, vars, thisType, false)
                    }
                    if (expType != null) {
                        val qc = QueryChecker(settings, ctx.query.text.removeSurrounding("\""), expType, ctx, "obj")
                        queryCheckers.add(qc)
                    }
                }
                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Construct_statementContext -> {
                var expType : Type? = null
                if(ctx.declType != null){
                    val lhs = ctx.expression(0)
                    if(lhs !is WhileParser.Var_expressionContext){
                        log("Variable declaration must declare a variable.", ctx)
                    } else {
                        val name = lhs.NAME().text
                        if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                        else {
                            expType = translateType(ctx.type(), className, generics)
                            vars[name] = expType
                        }
                    }
                }else{
                    expType = getType(ctx.target, inner, vars, thisType, false)
                }
                if(expType != null && fields.containsKey(expType.getPrimary().toString())) {
                    val fieldName = fields[expType.getPrimary().toString()]
                    for(f in fieldName!!.entries) {
                        val qc = QueryChecker(settings, ctx.query.text.removeSurrounding("\""), f.value.type, ctx, f.key)
                        queryCheckers.add(qc)
                    }
                }
                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Owl_statementContext -> {
                log("Type checking this form of (C)SSA is not supported yet ", ctx, Severity.WARNING)
                if(ctx.declType != null){
                    val lhs = ctx.expression(0)
                    if(lhs !is WhileParser.Var_expressionContext){
                        log("Variable declaration must declare a variable.", ctx)
                    } else {
                        val name = lhs.NAME().text
                        if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                        else vars[name] = translateType(ctx.type(), className, generics)
                    }
                }
                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Validate_statementContext -> {
                log("Type checking this form of (C)SSA is not supported yet ", ctx, Severity.WARNING)
                val inner = getType(ctx.query, inner, vars, thisType, inRule)
                if(inner != STRINGTYPE)
                    log("Validate expects a string to a SHACL shape file as its parameter",ctx)
                if(ctx.declType != null){
                    val lhs = ctx.expression(0)
                    if(lhs !is WhileParser.Var_expressionContext){
                        log("Variable declaration must declare a variable.", ctx)
                    } else {
                        val name = lhs.NAME().text
                        if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                        else vars[name] = translateType(ctx.type(), className, generics)
                    }
                }
                if(ctx.target != null && ctx.target !is WhileParser.Var_expressionContext && inRule)
                    log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Return_statementContext -> {
                val innerType = getType(ctx.expression(), inner, vars, thisType, inRule)
                if(innerType != ERRORTYPE && innerType != metType && !metType.isAssignable(innerType, extends))
                    log("Type $innerType of return statement does not match method type ${metType}.",ctx)
                return true
            }
            is WhileParser.Destroy_statementContext -> {
                val innerType = getType(ctx.expression(), inner, vars, thisType, inRule)
                if(innerType != ERRORTYPE && !OBJECTTYPE.isAssignable(innerType, extends))
                    log("Type $innerType of destroy statement is not an object type.",ctx)

                if(inRule) log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Output_statementContext -> {
                //For now, we print everything, so the important thing is just that it is not an error
                val innerType = getType(ctx.expression(), inner, vars, thisType, inRule)
                /*if(innerType == ERRORTYPE)
                    log("Println statement expects a string, but parameter has type $innerType.", ctx)*/
            }
            is WhileParser.Skip_statmentContext -> { }
            is WhileParser.Debug_statementContext -> { }
            is WhileParser.Scenario_statementContext -> {
                val path = ctx.path.text.removeSurrounding("\"")
                if (!Files.exists(Paths.get(path))) {
                    log("Could not find file for simulation scenario $path", ctx, Severity.WARNING)
                }

                if(ctx.type() != null) {
                    val declType = translateType(ctx.type(), className, generics)
                    if(declType != ScenarioType)
                        log("Scenarios require the Scenario type, got $declType", ctx)
                    if (ctx.target !is WhileParser.Var_expressionContext) {
                        log("Variable declaration must declare a variable.", ctx)
                    } else {
                        val name = ((ctx.target) as WhileParser.Var_expressionContext).NAME().text
                        if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                        else vars[name] = declType
                    }
                }else{
                    val declType = getType(ctx.target, inner, vars, thisType, false, read = false)
                    if(declType != ScenarioType)
                        log("Type $ScenarioType is not assignable to $declType", ctx)
                }
            }
            is WhileParser.Simulate_statementContext -> {
                val path = ctx.path.text.removeSurrounding("\"")
                if (!Files.exists(Paths.get(path))) {
                    log("Could not find file for FMU $path, statement cannot be type checked", ctx)
                } else {
                        val sim = Simulation(path)
                    val inits = if(ctx.varInitList() != null)
                        ctx.varInitList().varInit()
                            .associate { Pair(it.NAME().text, getType(it.expression(), inner, vars, thisType, inRule)) }
                    else
                                 emptyMap()

                    var ins = listOf<Pair<String,Type>>()
                    var outs = listOf<Pair<String,Type>>()

                    for(mVar in sim.modelDescription.modelVariables){
                        if(mVar.causality == "input" || mVar.causality == "state"){
                            if(!mVar.hasStartValue() && !inits.containsKey(mVar.name))
                                log("Simulation fails to initialize variable ${mVar.name}: no initial value given", ctx)
                        }
                        if((mVar.causality == "output" || mVar.initial == "calculated") && inits.containsKey(mVar.name)) {
                            log("Cannot initialize output or/and calculated variable ${mVar.name}",ctx)
                        }

                        if(mVar.causality == "input") ins = ins + Pair(mVar.name, getSimType(mVar.typeName))
                        if(mVar.causality == "output") outs = outs + Pair(mVar.name, getSimType(mVar.typeName))
                    }
                    val simType = SimulatorType(ins, outs)

                    if(ctx.type() != null) {
                        val declType = translateType(ctx.type(), className, generics)
                        if(!declType.isAssignable(simType, extends))
                            log("Type $simType is not assignable to $declType", ctx)
                        if (ctx.target !is WhileParser.Var_expressionContext) {
                            log("Variable declaration must declare a variable.", ctx)
                        } else {
                            val name = ((ctx.target) as WhileParser.Var_expressionContext).NAME().text
                            if (vars.keys.contains(name)) log("Variable $name declared twice.", ctx)
                            else vars[name] = translateType(ctx.type(), className, generics)
                        }
                    }else{
                        val declType = getType(ctx.target, inner, vars, thisType, false, read = false)
                        if(!declType.isAssignable(simType, extends))
                            log("Type $simType is not assignable to $declType", ctx)
                    }

                }
                if(inRule) log("Non-local access in rule method.", ctx)
            }
            is WhileParser.Tick_statementContext -> {
                if(inRule) log("Non-local access in rule method.", ctx)
                val fmuType = getType(ctx.fmu, inner, vars, thisType, inRule)
                val tickType = getType(ctx.time, inner, vars, thisType, inRule)
                if(fmuType !is SimulatorType)
                    log("Tick statement expects a FMU as first parameter, but got $fmuType }.",ctx)
                if(tickType != DOUBLETYPE)
                    log("Tick statement expects a Double as second parameter, but got $tickType }.",ctx)
            }
            else -> {
                log("Statements with class ${ctx.javaClass} cannot be type checked",ctx)
            }
        }
        return false
    }

    private fun getSimType(typeName: String): Type =
        when(typeName) {
            "Integer" -> INTTYPE
            "Real" -> DOUBLETYPE
            "String" -> STRINGTYPE
            "Boolean" -> BOOLEANTYPE
            else -> ERRORTYPE
        }


    // This cannot be done with extension methods because they cannot override ExpressionContext.getType()
    private fun getType(eCtx : WhileParser.ExpressionContext,
                        fields : Map<String, FieldInfo>,
                        vars : Map<String, Type>,
                        thisType : Type,
                        inRule : Boolean,
                        read: Boolean = true,
    ) : Type {
        when(eCtx){
            is WhileParser.Integer_expressionContext -> return INTTYPE
            is WhileParser.Double_expressionContext -> return DOUBLETYPE
            is WhileParser.String_expressionContext -> return STRINGTYPE
            is WhileParser.False_expressionContext -> return BOOLEANTYPE
            is WhileParser.True_expressionContext -> return BOOLEANTYPE
            is WhileParser.Null_expressionContext -> return NULLTYPE
            is WhileParser.Var_expressionContext -> {
                val name = eCtx.NAME().text
                if(!vars.containsKey(name) && !fields.containsKey(name)) log("Variable $name is not declared.", eCtx)
                if(!vars.containsKey(name) && fields.containsKey(name)) log("Variable $name is not declared. If you mean to use the field $name, use this.$name.", eCtx)
                return vars.getOrDefault(name, ERRORTYPE)
            }
            is WhileParser.Field_expressionContext -> {
                val name = eCtx.NAME().text
                if(!fields.containsKey(name))
                    log("Field $name is not declared for $thisType.", eCtx)
                return fields.getOrDefault(name, FieldInfo(eCtx.NAME().text, ERRORTYPE, Visibility.DEFAULT, thisType, false)).type
            }
            is WhileParser.Nested_expressionContext -> {
                return getType(eCtx.expression(), fields, vars, thisType, inRule)
            }
            is WhileParser.Mult_expressionContext -> { // The following has a lot of code duplication and could be improved by changing the grammar
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalFunction(t1, t2, "*", eCtx)
            }
            is WhileParser.Plus_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalFunction(t1, t2, "+", eCtx)
            }
            is WhileParser.Minus_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalFunction(t1, t2, "-", eCtx)
            }
            is WhileParser.Div_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalFunction(t1, t2, "/", eCtx)
            }
            is WhileParser.Mod_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                if(t2 != INTTYPE) log("Second parameter of % must be an integer, got $t2", ctx)
                if(t1 != INTTYPE && t1 != DOUBLETYPE) log("First parameter of % must be numeric, got $t1", ctx)
                return INTTYPE
            }
            is WhileParser.Neq_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                if((t1 == ERRORTYPE && t2 == ERRORTYPE) || (t2 == t1)) return BOOLEANTYPE
                if(t1.isAssignable(t2, extends) || t2.isAssignable(t1, extends)) return BOOLEANTYPE
                log("Malformed comparison <> with subtypes $t1 and $t2", eCtx)
                return ERRORTYPE
            }
            is WhileParser.Eq_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                if((t1 == ERRORTYPE && t2 == ERRORTYPE) || (t2 == t1)) return BOOLEANTYPE
                if(t1.isAssignable(t2, extends) || t2.isAssignable(t1, extends)) return BOOLEANTYPE
                log("Malformed comparison = with subtypes $t1 and $t2", eCtx)
                return ERRORTYPE
            }
            is WhileParser.And_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                if((t1 == ERRORTYPE && t2 == ERRORTYPE) || (t2 == BOOLEANTYPE && t1 == BOOLEANTYPE)) return BOOLEANTYPE
                if(t1.isAssignable(t2, extends) || t2.isAssignable(t1, extends)) return BOOLEANTYPE
                log("Malformed comparison && with subtypes $t1 and $t2", eCtx)
                return ERRORTYPE
            }
            is WhileParser.Or_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                if((t1 == ERRORTYPE && t2 == ERRORTYPE) || (t2 == BOOLEANTYPE && t1 == BOOLEANTYPE)) return BOOLEANTYPE
                if(t1.isAssignable(t2, extends) || t2.isAssignable(t1, extends)) return BOOLEANTYPE
                log("Malformed comparison || with subtypes $t1 and $t2", eCtx)
                return ERRORTYPE
            }
            is WhileParser.Not_expressionContext -> {
                val t1 = getType(eCtx.expression(), fields, vars, thisType, inRule)
                if(t1 == ERRORTYPE || t1 == BOOLEANTYPE) return BOOLEANTYPE
                log("Malformed negation ! with subtypes $t1", eCtx)
                return ERRORTYPE
            }
            is WhileParser.Leq_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalRelation(t1, t2, "<=", eCtx)
            }
            is WhileParser.Geq_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalRelation(t1, t2, ">=", eCtx)
            }
            is WhileParser.Lt_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalRelation(t1, t2, "<", eCtx)
            }
            is WhileParser.Gt_expressionContext -> {
                val t1 = getType(eCtx.expression(0), fields, vars, thisType, inRule)
                val t2 = getType(eCtx.expression(1), fields, vars, thisType, inRule)
                return typeForNumericalRelation(t1, t2, ">", eCtx)
            }
            is WhileParser.This_expressionContext -> return thisType
            is WhileParser.Fmu_field_expressionContext -> {
                val t1 = getType(eCtx.expression(), fields, vars, thisType, inRule)
                if(t1 == ERRORTYPE) return ERRORTYPE
                if(t1 !is SimulatorType) {
                    log("Port method is specific to FMU objects.", eCtx)
                    return ERRORTYPE
                }
                val name = eCtx.STRING().text.removeSurrounding("\"")
                return if(read){
                    val inVar = t1.outVar.firstOrNull { it.first == name }
                    if(inVar == null)
                        log("Trying to read from a field that is not an outport: $name", eCtx)
                    inVar?.second ?: ERRORTYPE
                } else {
                    val inVar = t1.inVar.firstOrNull { it.first == name}
                    if(inVar == null)
                        log("Trying to write into a field that is not an inport: $name", eCtx)
                    inVar?.second ?: ERRORTYPE
                }
            }
            is WhileParser.External_field_expressionContext -> { // This must resolve the generics
                val t1 = getType(eCtx.expression(), fields, vars, thisType, inRule)
                if(t1 == ERRORTYPE) return ERRORTYPE
                if(t1 is SimulatorType){
                    if(eCtx.NAME().text == SimulatorObject.ROLEFIELDNAME)
                        return STRINGTYPE
                    if(eCtx.NAME().text == SimulatorObject.PSEUDOOFFSETFIELDNAME)
                        return DOUBLETYPE
                    if(eCtx.NAME().text == SimulatorObject.TIMEFIELDNAME && read)
                        return DOUBLETYPE
                    if(eCtx.NAME().text == SimulatorObject.TIMEFIELDNAME && !read) {
                        log("Trying to write the time of an FMU object. Do you mean ${SimulatorObject.PSEUDOOFFSETFIELDNAME}? ${eCtx.NAME().text}", eCtx)
                        return DOUBLETYPE
                    }
                    return if(read){
                        val inVar = t1.outVar.firstOrNull { it.first == eCtx.NAME().text }
                        if(inVar == null)
                            log("Trying to read from a field that is not an outport: ${eCtx.NAME().text}", eCtx)
                        inVar?.second ?: ERRORTYPE
                    } else {
                        val inVar = t1.inVar.firstOrNull { it.first == eCtx.NAME().text }
                        if(inVar == null)
                            log("Trying to write into a field that is not an inport: ${eCtx.NAME().text}", eCtx)
                        inVar?.second ?: ERRORTYPE
                    }
                }
                if(t1 is GenericType) {
                    log("Access of fields of generic types is not supported.", eCtx)
                    return ERRORTYPE
                }
                val primary = t1.getPrimary()
                val primName = primary.getNameString()
                if(!this.fields.containsKey(primName)){
                    log("Cannot access fields of $primary.", eCtx)
                    return ERRORTYPE
                }
                val otherFields = getFields(primName)
                if(!otherFields.containsKey(eCtx.NAME().text)){
                    log("Field ${eCtx.NAME().text} is not declared for $primary.", eCtx)
                    return ERRORTYPE
                } else {
                    if(inRule && thisType != otherFields[eCtx.NAME().text]!!.declaredIn)
                        log("Field ${eCtx.NAME().text} accessed in rule-method.", eCtx)
                    if(inRule && !otherFields[eCtx.NAME().text]!!.declaredIn.isAssignable(thisType, extends))
                        log("Field ${eCtx.NAME().text} accessed in rule-method.", eCtx)
                }
                val fieldType = this.fields.getOrDefault(primary.getNameString(), mutableMapOf()).getOrDefault(eCtx.NAME().text,
                    FieldInfo(eCtx.NAME().text, ERRORTYPE, Visibility.DEFAULT, thisType, false)
                )
                return instantiateGenerics(fieldType.type, t1, primName, generics.getOrDefault(thisType.getPrimary().getNameString(), listOf()))
            }
            else -> {
                log("Expression $eCtx cannot be type checked.",eCtx)
                return ERRORTYPE
            }
        }
    }


    private fun typeForNumericalFunction(t1 : Type, t2 : Type, symbol:String, ctx: ParserRuleContext) : Type {
        if(t1 == INTTYPE && t2 == INTTYPE) return INTTYPE
        if(t1 == DOUBLETYPE && t2 == DOUBLETYPE) return DOUBLETYPE
        if(t1 == INTTYPE && t2 == DOUBLETYPE) return DOUBLETYPE
        if(t1 == DOUBLETYPE && t2 == INTTYPE) return DOUBLETYPE
        if(t1 == ERRORTYPE && (t2 == INTTYPE || t2 == DOUBLETYPE)) return t2
        if(t2 == ERRORTYPE && (t1 == INTTYPE || t1 == DOUBLETYPE)) return t2
        log("Malformed operator $symbol with subtypes $t1 and $t2", ctx)
        return ERRORTYPE
    }

    private fun typeForNumericalRelation(t1 : Type, t2 : Type, symbol:String, ctx: ParserRuleContext) : Type {
        if((t1 == INTTYPE || t1 == DOUBLETYPE || t1 == ERRORTYPE) && (t2 == INTTYPE || t2 == DOUBLETYPE || t2 == ERRORTYPE)) return BOOLEANTYPE
        if(    (INTTYPE.isAssignable(t1, extends) || DOUBLETYPE.isAssignable(t1, extends))
            && (INTTYPE.isAssignable(t2, extends) || DOUBLETYPE.isAssignable(t2, extends))) return BOOLEANTYPE
        log("Malformed comparison $symbol with subtypes $t1 and $t2", ctx)
        return ERRORTYPE
    }

    /**********************************************************************
    Helper to handle types
     ***********************************************************************/
    private fun getParameterTypes(met: WhileParser.Method_defContext, otherClassName: String): List<Type> =
        if(met.paramList() == null) listOf() else met.paramList().param().map { translateType(it.type(), otherClassName, generics) }


    private fun getParameterTypes(className: String): List<Type> {
        return fields.getOrDefault(className, mapOf()).map { it.value.type }
    }


    private fun getMethods(className: String): List<WhileParser.Method_defContext> {
        val ret : List<WhileParser.Method_defContext> = methods.getOrDefault(className, listOf())
        if(extends.containsKey(className)){
            val supertype = extends.getOrDefault(className, ERRORTYPE).getPrimary().getNameString()
            val more = getMethods(supertype)
            return more + ret
        }
        return ret
    }

    /* check whether @type contains an unknown (structural) subtype  */
    private fun containsUnknown(type: Type, types: Set<String>): Boolean =
        when(type){
        is GenericType -> false // If we can translate it into a generic type, we already checked
        is SimulatorType -> false
        is BaseType -> !types.contains(type.name)
        else -> {
            val composType = type as ComposedType
            composType.params.fold(
                containsUnknown(composType.name, types)
            ) { acc, nx -> acc || containsUnknown(nx, types) }
        }
    }

    private fun getFields(className: String): Map<String, FieldInfo> {
        val f = fields.getOrDefault(className, mapOf())
        if(extends.containsKey(className)){
            val supertype = extends.getOrDefault(className, ERRORTYPE).getPrimary().getNameString()
            val moreFields = getFields(supertype)
            return moreFields + f
        }
        return f
    }

    private fun getLeftoverAbstract(clCtx : WhileParser.Class_defContext) : List<String> {
        val newAbs = clCtx.method_def().filter { it.abs != null }.map { it.NAME().text }
        if(clCtx.superType != null){
            val over = recoverDef[extends[clCtx.NAME().text]?.getPrimary()?.getNameString()] ?: return newAbs
            val above = getLeftoverAbstract(over)
            val impl = clCtx.method_def().filter { it.abs == null }.map { it.NAME().text }
            return (above - impl.toSet()) + newAbs
        } else {
            return newAbs
        }

    }
    /**********************************************************************
    Helper to handle generics
     ***********************************************************************/

    /*
        schablone is the type of the part of className that we focus on
        concrete  is the type of the relevant className instance
        className is the class whose definitions binds the generics
        gens      are the generics of the focused class
     */
    private fun instantiateGenerics(schablone: Type, concrete: Type, className: String, gens : List<String>): Type {
        //get definition of class that binds generics
        val otherClass = recoverDef[className]

        //get its type
        val otherAbstractType =
            if(otherClass!!.namelist() == null) BaseType(className)
            else ComposedType(BaseType(className), otherClass.namelist().NAME().map { GenericType(it.text) })

        //match type of class with the concrete instance
        val matching = match(otherAbstractType, concrete)

        //apply retrieved matching to abstract part
        return applyMatching(schablone, matching, gens)
    }

    /* apply unifier */
    private fun applyMatching(metRet: Type, matching: Map<GenericType, Type>, gens : List<String>) : Type {
        when(metRet){
            is GenericType -> {
                return when {
                    matching.containsKey(metRet) -> matching.getOrDefault(metRet, ERRORTYPE)
                    gens.contains(metRet.name) -> metRet
                    else -> {
                        log("Applying the matching on $metRet failed.", null)
                        ERRORTYPE
                    }
                }
            }
            is BaseType -> return metRet
            is SimulatorType -> return metRet
            is ComposedType -> {
                return ComposedType(
                    applyMatching(metRet.name, matching, gens),
                    metRet.params.map { applyMatching(it, matching, gens) })
            }
            else -> return ERRORTYPE //not needed, but type system fails otherwise
        }
    }

    /* compute unifier */
    private fun match(abstractType: Type, concreteType: Type): Map<GenericType, Type> {
        when(abstractType){
            is GenericType -> return mapOf(Pair(abstractType, concreteType))
            is BaseType -> return mapOf()
            is ComposedType -> {
                if(concreteType !is ComposedType){
                    log("Matching $abstractType with $concreteType failed.", null)
                } else {
                    var matching = match(abstractType.name, concreteType.name)
                    if(abstractType.params.size != concreteType.params.size){
                        log("Matching $abstractType with $concreteType failed.", null)
                    } else {
                        for (i in abstractType.params.indices){
                            val next = match(abstractType.params[i], concreteType.params[i])
                            for(pair in next){
                                if(matching.containsKey(pair.key)){
                                    log("Matching $abstractType with $concreteType failed.", null)
                                }else matching = matching + Pair(pair.key, pair.value)
                            }
                        }
                        return matching
                    }
                }
                return mapOf()
            }
            else -> return mapOf()
        }
    }

    private fun getClassDecl(ctx : RuleContext): WhileParser.Class_defContext? {
        if(ctx.parent == null) return null
        if(ctx.parent is WhileParser.Class_defContext) return ctx.parent as WhileParser.Class_defContext
        return getClassDecl(ctx.parent)
    }

    private fun getGenericAssignment(ctx : WhileParser.Class_defContext): Map<GenericType, Type> {
        if(ctx.superType == null) return emptyMap()
        val superType = translateType(ctx.superType, ctx.className.text, generics) //instantianted
        val superName = superType.getPrimary().getNameString()
        val superDecl = recoverDef[superName] ?: return emptyMap() //declaration
        if(superType !is ComposedType) return getGenericAssignment(superDecl)
        val absType =
            if(superDecl.namelist() == null) BaseType(superName)
            else ComposedType(BaseType(superName), superDecl.namelist().NAME().map { GenericType(it.text) })
        return match(absType, superType)
    }
}
