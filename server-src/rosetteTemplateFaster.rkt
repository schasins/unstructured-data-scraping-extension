#lang s-exp rosette

; Get the dataset
(require (planet neil/csv:1:=7))
(require (planet williams/describe/describe))
(require rosette/lib/meta/meta)

(define csvLs (csv->list (open-input-file "trainingSetSingeNodeFeatureVectors.csv")))

(define bitwidth (string->number (car (car csvLs)))) ; first row of the dataset is just the bitwidth we need
(define datasetRaw (cdr csvLs))
(define columnLabels (car datasetRaw))
(define columnTypes (cadr datasetRaw))
(define oldMins (caddr datasetRaw))
(define oldMaxes (cadddr datasetRaw))
(define remainder (cddddr datasetRaw))
(define newMins (car remainder))
(define newMaxes (cadr remainder))
(define datasetOnly (cddr remainder))

; let's make things numeric
(define dataset (map (lambda (row)
                       (cons (car row) (cons (cadr row) (map (lambda (cell) (string->number cell)) (cddr row))))) ; first two cells are label and document name
                     datasetOnly))

(define numRows (length datasetOnly))
(define numNonNoLabelRows
	(foldl (lambda (row acc)
	       (if (equal? (list-ref row 0) "nolabel") acc (+ acc 1))) 0 dataset))
(define targetNumNoLabelRowsToFilter (- numRows (* 4 numNonNoLabelRows))) ; should have at most 3 nolabel rows to every non-nolabel row

(current-bitwidth bitwidth)

; let's divide the dataset into the ones we want to caputre and the ones we don't

(define labelRows
  (filter (lambda (row) (not (equal? (list-ref row 0) "nolabel"))) dataset))

(define nolabelRows
  (filter (lambda (row) (equal? (list-ref row 0) "nolabel")) dataset))

; Score calculations


(define (avoidsLabeledData pred)
  (andmap (lambda (row) (not (pred row))) labelRows))

(define (findsManyNoLabels pred)
  (>=
   (foldl (lambda (row count) (if (pred row) (+ count 1) count)) 0 nolabelRows)
   targetNumNoLabelRowsToFilter)
  )


(define goodFilter (lambda (filter)
                     (begin (assert (avoidsLabeledData filter)) (assert (findsManyNoLabels filter)))))

; Actual synthesis

(define-synthax (simpleFilterGrammar datasetRow)
  [choose
  ([choose < >] (list-ref datasetRow
                          [choose ###indexesForNumericCols###]
                 )
                [choose ###numericalComparisonValues###])
  ]
  )

; A sketch with a hole to be filled with an expr
(define (filterSynthesized row) (simpleFilterGrammar row))
(define start (current-seconds))
(define synthesizedFilter
   (synthesize #:forall '()
    #:guarantee (goodFilter filterSynthesized)))
(define end (current-seconds))

(print-forms synthesizedFilter)
(- end start)





