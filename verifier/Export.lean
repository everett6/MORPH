import Mathlib.Data.Nat.Basic
import Mathlib.Logic.Basic
import Lean
open Lean Elab Command Meta
run_cmd do
  let env ← getEnv
  for (name, info) in env.constants.toList do
    if info.isTheorem && info.levelParams.isEmpty && !name.toString.startsWith "_" then
      let type ← liftTermElabM do return (← ppExpr info.type).pretty
      let record := Json.mkObj [("name", toJson name.toString), ("statement", toJson type)]
      liftIO <| IO.println record.compress
