; ModuleID = 'probe2.89300a67-cgu.0'
source_filename = "probe2.89300a67-cgu.0"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx11.0.0"

; probe2::probe
; Function Attrs: uwtable
define void @_ZN6probe25probe17hdee6b071b9dbbbe7E() unnamed_addr #0 {
start:
  %0 = alloca i32, align 4
  store i32 -2147483648, i32* %0, align 4
  %1 = load i32, i32* %0, align 4
  br label %bb1

bb1:                                              ; preds = %start
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.bitreverse.i32(i32) #1

attributes #0 = { uwtable "frame-pointer"="non-leaf" "target-cpu"="apple-a14" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 2}
