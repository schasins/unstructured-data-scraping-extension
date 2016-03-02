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
(define targetNumNoLabelRowsToFilter (- numRows (* 3 numNonNoLabelRows))) ; should have at most 2 nolabel rows to every non-nolabel row

(current-bitwidth bitwidth)

; Score calculations
; score is tuple of form: (number of nolabel items in filter, number of non-nolabel items in filter)

(define score (lambda (pred dataset)
  (foldl (lambda (row tuple)
           (begin
             ;(print (pred row))
           (if (pred row)
               (begin ;(print "t") (newline)
               (if (equal? (list-ref row 0) "nolabel") 
                   `( ,(+ (list-ref tuple 0) 1) ,(list-ref tuple 1))
                   `( ,(list-ref tuple 0) ,(+ (list-ref tuple 1) 1))))
               
               (begin ;(print "f") (newline)
                 tuple)
               )))
         '(0 0) dataset)))

(define goodFilter (lambda (filter dataset)
	(let ([scoreTuple (score filter dataset)])
	     (begin (assert (> (list-ref scoreTuple 0) targetNumNoLabelRowsToFilter) )
	     	  (assert (= (list-ref scoreTuple 1) 0))
                  ))))

; Actual synthesis

(define-synthax (simpleFilterGrammar datasetRow)
  [choose
  ([choose < >] (list-ref datasetRow [choose ###indexesForBooleanCols###]) 0)
  ([choose < >] (list-ref datasetRow [choose ###indexesForNumericCols###]) (??))
  ]
  )

; A sketch with a hole to be filled with an expr
(define (filterSynthesized row) (simpleFilterGrammar row))
(define synthesizedFilter
   (synthesize #:forall '()
    #:guarantee (goodFilter filterSynthesized dataset)))

(print-forms synthesizedFilter)






