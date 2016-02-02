#lang s-exp rosette

; Get the dataset

(require (planet neil/csv:1:=7))
(require (planet williams/describe/describe))
(require rosette/lib/meta/meta)

(define datasetRaw (csv->list (open-input-file "sample.csv")))

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

(define numFeatures (- (length (car datasetOnly)) 2)) ; - 2 because label, doc name at start of row
(define numRows (length datasetOnly))
(define numNonNoLabelRows
	(foldl (lambda (row acc)
	       (if (equal? (list-ref row 0) "nolabel") acc (+ acc 1))) 0 dataset))
(define targetNumNoLabelRowsToFilter (- numRows (* 3 numNonNoLabelRows))) ; should have at most 2 nolabel rows to every non-nolabel row

(current-bitwidth 10) ;TODO: set bitwdith according to user-given limit

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
  (> (list-ref datasetRow [choose 3 4 5 6 7 8 14]) 0)
  (> (list-ref datasetRow [choose 300 301 302 303 304 305 306 307 308]) (??))
  ]
  )

#;(define-symbolic c number?)
#;(define-synthax (simpleFilterGrammar datasetRow colIndexes ...)
  (let ( [colIndex [choose colIndexes ...]]) (> (list-ref datasetRow colIndex) c)))


(define-symbolic n number?)
(define sol
    (solve (begin (assert (> n 0))
                  (assert (< (add1 n) 0)))))
(evaluate n sol) 

(define-symbolic x number?)

; A sketch with a hole to be filled with an expr
(define (filterSynthesized row) (simpleFilterGrammar row))
; todo: put the proper numbers in instead of testcolumnindex
(define synthesizedFilter
   (synthesize #:forall (list x)
    #:guarantee (goodFilter filterSynthesized dataset)))

(print-forms synthesizedFilter)

(define (filterSynthesizedAns row) (> (list-ref row 302) 156))
(goodFilter filterSynthesizedAns dataset)






