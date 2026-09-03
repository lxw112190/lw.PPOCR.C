package com.lxw112190.ppocr.demo

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.lxw112190.ppocr.OcrLine
import java.util.Locale

class OcrResultAdapter(
    private val onLineClicked: (Int) -> Unit,
) : ListAdapter<OcrLine, OcrResultAdapter.LineViewHolder>(DIFF_CALLBACK) {
    private var selectedIndex: Int? = null

    fun setSelectedIndex(index: Int?) {
        selectedIndex = index
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LineViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_ocr_line, parent, false)
        return LineViewHolder(view)
    }

    override fun onBindViewHolder(holder: LineViewHolder, position: Int) {
        holder.bind(getItem(position), selectedIndex)
        holder.itemView.setOnClickListener {
            val adapterPosition = holder.bindingAdapterPosition
            if (adapterPosition != RecyclerView.NO_POSITION) {
                selectedIndex = getItem(adapterPosition).index
                notifyDataSetChanged()
                onLineClicked(selectedIndex!!)
            }
        }
    }

    class LineViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val index: TextView = itemView.findViewById(R.id.line_index)
        private val text: TextView = itemView.findViewById(R.id.line_text)
        private val scores: TextView = itemView.findViewById(R.id.line_scores)

        fun bind(line: OcrLine, selectedIndex: Int?) {
            index.text = "%02d".format(line.index + 1)
            text.text = line.text
            scores.text = String.format(
                Locale.US,
                "检测 %.1f%% · 识别 %.1f%%",
                line.detScore * 100f,
                line.recScore * 100f,
            )
            itemView.setBackgroundColor(
                if (selectedIndex == line.index) Color.rgb(231, 240, 251)
                else Color.TRANSPARENT,
            )
        }
    }

    companion object {
        private val DIFF_CALLBACK = object : DiffUtil.ItemCallback<OcrLine>() {
            override fun areItemsTheSame(oldItem: OcrLine, newItem: OcrLine): Boolean =
                oldItem.index == newItem.index

            override fun areContentsTheSame(oldItem: OcrLine, newItem: OcrLine): Boolean =
                oldItem.index == newItem.index &&
                    oldItem.text == newItem.text &&
                    oldItem.detScore == newItem.detScore &&
                    oldItem.recScore == newItem.recScore &&
                    oldItem.box.contentEquals(newItem.box)
        }
    }
}
